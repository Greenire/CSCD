import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem

from ...utils.config import Config
from ..base import BaseOptimizer
from ..conformers import ConformerGenerator
from ..jobs import JobDispatcher

class XTBOptimizer(BaseOptimizer):
    """XTB-based molecular descriptor calculator.
    
    Calculates molecular descriptors using XTB quantum chemistry package.
    Supports geometry optimization and single-point energy calculations.
    Calculations can be managed by various job scheduling systems.
    
    Attributes:
        STORAGE_METHOD (str): Always 'multi' as XTB calculations are stored in directories
        DEFAULT_PARAMS (dict): Default parameters for XTB calculations
    """
    
    STORAGE_METHOD = 'multi'
    
    DEFAULT_PARAMS = {
        'method': 'gfn2',      # GFN2-xTB method
        'opt': 'extreme',      # Optimization level
        'charge': 0,           # Molecular charge
        'solvent': 'none',     # Solvent model
        'max_workers': None    # Number of parallel workers (None = CPU count)
    }
    
    def __init__(
        self,
        raw_data_name: str,
        species_name: str,
        config: Optional[Config] = None
    ) -> None:
        """Initialize XTB calculator."""
        super().__init__(raw_data_name, species_name, config=config)
        self._setup_calculation_params()
        
        # Initialize job dispatcher
        self._job_dispatcher = JobDispatcher(self.config)
        
        # Initialize conformer generator
        self._conf_generator = ConformerGenerator(self.config)
        
    def _setup_calculation_params(self) -> None:
        """Setup XTB calculation parameters from config."""
        self.params = self.DEFAULT_PARAMS.copy()
        
        # Update from config if provided
        xtb_config = self.config.get('molecular_descriptors.xtb', {})
        self.params.update({
            'method': xtb_config.get('method', self.params['method']),
            'opt': xtb_config.get('optimize', self.params['opt']),
            'charge': xtb_config.get('charge', self.params['charge']),
            'solvent': xtb_config.get('solvent', self.params['solvent']),
            'max_workers': self.config.get('job_manager.max_workers', None)  # None = CPU count
        })

        # Build XTB command with f-strings
        charge_param = f" --charge {self.params['charge']}" if self.params['charge'] != 0 else ""
        solvent_param = f" --alpb {self.params['solvent']}" if self.params['solvent'] != 'none' else ""
        self._command = f"xtb --{self.params['method']} --opt {self.params['opt']} -P {self.params['max_workers']}{charge_param}{solvent_param}"
            
        self._logger.info(f"XTB parameters: {self.params}")
        self._logger.info(f"XTB command template: {self._command}")

    def _check_optimization_finished(self, calc_dir: Path) -> bool:
        """Check if optimization is finished and successful."""
        err_log = calc_dir / 'stderr.log'
        if not err_log.exists():
            return False
            
        return 'normal termination of xtb' in err_log.read_text()

    def _write_xyz(self, mol: Chem.Mol, conf_id: int, xyz_file: Path) -> None:
        """Write molecule coordinates in XYZ format."""
        Chem.MolToXYZFile(mol, str(xyz_file), confId=conf_id)

    def _setup_calculation_batch(self, batch_data, is_conformer: bool, n_conformers: int, overwrite: bool, kwargs: dict):
        """Setup calculations for a batch of molecules.
        
        Args:
            batch_data: Tuple of (idx, smiles)
            is_conformer: Whether to generate conformers
            n_conformers: Number of conformers to generate
            overwrite: Whether to overwrite existing results
            kwargs: Additional arguments for calculation setup
            
        Returns:
            List of job specifications
        """
        idx, smiles = batch_data


        mol, top_k_conf_ids = self._conf_generator.generate(smiles, num_top=n_conformers)
        if not top_k_conf_ids:
            raise RuntimeError(f"Failed to generate conformers for {smiles}")

        if is_conformer:
            return [
                calc for calc in [
                    self._setup_calculation(idx, mol, conf_id=i, top_k_conf_ids=top_k_conf_ids, overwrite=overwrite, **kwargs)
                    for i in range(n_conformers)
                ] if calc is not None
            ]
        calc = self._setup_calculation(idx, mol, top_k_conf_ids=top_k_conf_ids, overwrite=overwrite, **kwargs)
        return [calc] if calc is not None else []

    def _setup_calculation(
        self,
        idx: int,
        mol: Chem.Mol,
        conf_id: Optional[int] = None,
        top_k_conf_ids: Optional[List[int]] = None,
        overwrite: bool = False,
        **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Setup a single XTB calculation.
        
        Args:
            idx: Molecule index
            smiles: SMILES string
            conf_id: Conformer ID if applicable
            overwrite: Whether to overwrite existing results
            
        Returns:
            Dictionary with calculation settings or None if setup fails
        """
        # Create calculation directory
        if conf_id is None:
            calc_dir = self.results_file / str(idx)
            conf_id = 0
        else:
            calc_dir = self.results_file / f"{idx}_{conf_id}"
            
        if overwrite and calc_dir.exists():
            self._logger.info(f"Overwriting existing results in {calc_dir}")
            import shutil
            shutil.rmtree(calc_dir)
            
        if self._check_optimization_finished(calc_dir):
            self._logger.info(f"Results already exist in {calc_dir}")
            return None
            
        calc_dir.mkdir(parents=True, exist_ok=True)
        
        # Write XYZ file
        xyz_file = calc_dir / 'molecule.xyz'
        self._write_xyz(mol, top_k_conf_ids[conf_id], xyz_file)
        
        # Create job specification
        return {
            'command': self._command,
            'args': ['molecule.xyz'],
            'job_path': str(calc_dir)
        }


    def calculate(
        self,
        smiles: Union[str, List[str]],
        is_conformer: bool = False,
        overwrite: bool = False,
        **kwargs: Any
    ) -> None:
        """Start XTB calculations for molecules.
        
        This method submits calculations to the job dispatcher and returns immediately.
        Use check_calculations() to monitor progress and get_results() to retrieve results.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            is_conformer: Whether to generate conformers
            overwrite: Whether to overwrite existing results
            **kwargs: Additional arguments passed to XTB
        """
        # Get molecules to calculate
        smiles_list = [smiles] if isinstance(smiles, str) else smiles

        to_calculate, existing_results = self.check_calculations(
            smiles_list, is_conformer, overwrite
        )
        if not to_calculate:
            self._logger.info("No molecules to calculate")
            return
            
        self._logger.info(f"Calculating {len(to_calculate)} molecules")
            
        n_required_conformers = self.config.get('molecular_descriptors.conformer.n_conformers', 5)
        max_workers = self.config.get('molecular_descriptors.xtb.max_workers', 8)  # None = CPU count
        
        # Create a partial function with fixed arguments
        setup_func = partial(
            self._setup_calculation_batch,
            is_conformer=is_conformer,
            n_conformers=n_required_conformers,
            overwrite=overwrite,
            kwargs=kwargs
        )
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and gather results
            future_to_batch = {
                executor.submit(setup_func, (idx, smiles)): (idx, smiles)
                for idx, smiles in to_calculate.items()
            }
            
            # Collect results as they complete
            job_specs = []
            for future in future_to_batch:
                try:
                    specs = future.result()
                    if specs:
                        job_specs.extend(specs)
                except Exception as e:
                    self._logger.error(f"Error setting up calculation: {str(e)}")
                    
        if not job_specs:
            self._logger.warning("No valid job specifications created")
            return
            
        # Submit jobs through dispatcher
        batch_size = self.config.get('molecular_descriptors.xtb.batch_size', 500)
        self._logger.info(f"Submitting {len(job_specs)} calculations in batches of {batch_size}")
        
        job_ids = self._job_dispatcher.submit_jobs(job_specs, batch_size=batch_size)
        
        # Save job IDs for status tracking
        job_info = {
            'job_ids': job_ids,
            'specs': job_specs
        }
        job_info_file = self.opt_dir / 'job_info.json'
        with open(job_info_file, 'w') as f:
            json.dump(job_info, f, indent=2)
            
        self._logger.info(f"Submitted {len(job_ids)} jobs")
        
    def get_calculation_status(self) -> Dict[str, int]:
        """Get status of all running calculations.
        
        Returns:
            Dictionary with counts for each status:
                pending: Jobs waiting to start
                running: Currently running jobs
                completed: Successfully completed jobs
                failed: Failed jobs
        """
        status = {
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0
        }
        
        # Load job information
        job_info_file = self.opt_dir / 'job_info.json'
        if not job_info_file.exists():
            return status
            
        with open(job_info_file) as f:
            job_info = json.load(f)
            
        # Check job status through dispatcher
        job_status = self._job_dispatcher.check_status(job_info['job_ids'])
        
        # Map states to our status categories
        for state in job_status.values():
            if state in ['PENDING', 'REQUEUED']:
                status['pending'] += 1
            elif state in ['RUNNING', 'COMPLETING']:
                status['running'] += 1
            elif state in ['COMPLETED', 'DONE']:
                status['completed'] += 1
            else:
                status['failed'] += 1
                
        return status
        
    def get_results(
        self,
        wait: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get results from completed calculations.
        
        Args:
            wait: If True, wait for all calculations to complete
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary with calculation results
        """
        # Load job information
        job_info_file = self.opt_dir / 'job_info.json'
        if not job_info_file.exists():
            return {}
            
        with open(job_info_file) as f:
            job_info = json.load(f)
            
        # Wait for completion if requested
        if wait:
            start_time = time.time()
            while True:
                status = self.get_calculation_status()
                if status['pending'] == 0 and status['running'] == 0:
                    break
                    
                if timeout and time.time() - start_time > timeout:
                    self._logger.warning("Timeout waiting for calculations")
                    break
                    
                time.sleep(10)
                
        # Get results through dispatcher
        job_results = self._job_dispatcher.get_results(job_info['job_ids'])
        return self._process_results(job_results)

    def _process_results(self, job_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process XTB calculation results.
        
        Args:
            job_results: Raw job results from dispatcher
            
        Returns:
            Dictionary with processed results
        """
        results = {}
        for job_id, status in job_results.items():
            if isinstance(status, dict) and 'error' in status:
                self._logger.error(f"Job {job_id} failed: {status['error']}")
                continue
                
            # Get calculation directory from job info
            job_info_file = self.opt_dir / 'job_info.json'
            with open(job_info_file) as f:
                job_info = json.load(f)
                
            job_spec = next(
                spec for spec in job_info['specs'] 
                if str(id(spec)) == job_id
            )
            calc_dir = Path(job_spec['job_path'])
            
            # Check if optimization finished successfully
            if not self._check_optimization_finished(calc_dir):
                self._logger.error(f"Optimization failed in {calc_dir}")
                continue
                
            # Read optimized structure and energy
            try:
                # Convert XYZ to SDF if not already done
                if not (calc_dir / 'xtbopt.sdf').exists():
                    import os
                    os.system(f"cd {calc_dir} && obabel xtbopt.xyz -ixyz -osdf -O xtbopt.sdf")
                    
                # Read results
                with open(calc_dir / 'energy') as f:
                    energy = float(f.read().strip())
                    
                results[job_id] = {
                    'energy': energy,
                    'structure': str(calc_dir / 'xtbopt.sdf')
                }
                
            except Exception as e:
                self._logger.error(f"Error processing results in {calc_dir}: {str(e)}")
                continue
                
        return results
