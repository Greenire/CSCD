from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..utils.config import Config, get_config
from ..utils.logger import log_data_operation, setup_logger
from .base import BaseJobManager

class JobDispatcher:
    """Dispatcher for selecting and managing job scheduling systems"""
    
    # Map job manager types to their implementations
    _MANAGERS = {
        'slurm': 'SlurmJobManager',
        'pbs': 'PBSJobManager',
        'lsf': 'LSFJobManager',
        'local': None
    }
    
    def __init__(self, config: Optional[Config] = None) -> None: # type: ignore[override] # (Config):
        """
        Initialize job dispatcher
        
        Args:
            config: Configuration with settings:
                type: Type of job manager ('slurm', 'pbs', 'lsf', 'local')
                [type]_params: Parameters for specific job manager
                templates_dir: Directory with job templates
                max_jobs: Maximum concurrent jobs
                max_array_size: Maximum array job size
        """
        self.config = config or get_config()
        self._logger = setup_logger(__name__)
        self._current_specs = []
        
        # Get manager type
        self.manager_type = config.get('job_manager.type', 'local').lower()
        if self.manager_type not in self._MANAGERS:
            self._logger.warning(
                f"Unknown job manager type: {self.manager_type}, falling back to local"
            )
            self.manager_type = 'local'
            
        # Initialize manager
        self._init_manager()
        
    def _init_manager(self) -> None:
        """Initialize the selected job manager"""
        manager_class = self._MANAGERS[self.manager_type]
        
        if self.manager_type == 'local':
            # For local execution, we don't need ThreadPoolExecutor anymore
            self._manager = None
        else:
            # Import manager class dynamically
            module = __import__(__name__, fromlist=[manager_class])
            ManagerClass = getattr(module, manager_class)
            self._manager = ManagerClass(self.config)
            
    def submit_jobs(self, job_specs: List[Dict[str, Any]], 
                   batch_size: int = 500,
                   wait: bool = True,
                   poll_interval: int = 60) -> Union[List[str], Dict[str, str]]:
        """
        Submit jobs using the selected manager
        
        Args:
            job_specs: List of job specifications
            batch_size: Size of job batches for array jobs
            wait: Whether to wait for job completion
            poll_interval: Time in seconds between status checks
            
        Returns:
            If wait=True: Dictionary mapping job IDs to final status
            If wait=False: List of job IDs
        """
        if self.manager_type == 'local':
            # For local execution, run jobs sequentially
            job_ids = []
            pwd = os.getcwd()
            self._current_specs = job_specs
            
            for spec in job_specs:
                try:
                    job_path = spec['job_path']
                    os.chdir(job_path)
                    
                    # Create command string
                    cmd = spec['command']
                    if 'args' in spec:
                        cmd = f"{cmd} {' '.join(map(str, spec['args']))}"
                        
                    # Run command and wait for completion
                    self._logger.info(f"Running job in {job_path}: {cmd}")
                    with open('stdout.log', 'w') as stdout_file:
                        with open('stderr.log', 'w') as stderr_file:
                            result = os.system(f"{cmd} > stdout.log 2> stderr.log")
                            
                    if result != 0:
                        self._logger.error(f"Job failed with exit code {result}")
                    
                    job_ids.append(str(id(spec)))
                    
                except Exception as e:
                    self._logger.error(f"Error running job: {str(e)}")
                    job_ids.append(str(id(spec)))  # Still add ID for tracking
                finally:
                    os.chdir(pwd)
                    
            return {job_id: 'COMPLETED' for job_id in job_ids} if wait else job_ids
        else:
            # Submit jobs through manager
            job_ids = self._manager.submit_jobs(job_specs, batch_size)
            self._logger.info(f"Submitted {len(job_ids)} jobs to {self.manager_type}")
            
            # Wait for completion if requested
            if wait:
                return self.wait_for_completion(job_ids, poll_interval)
            return job_ids
        
    def wait_for_completion(self, job_ids: List[str], poll_interval: int = 60) -> Dict[str, str]:
        """
        Wait for jobs to complete while monitoring their progress
        
        Args:
            job_ids: List of job IDs to monitor
            poll_interval: Time in seconds between status checks
            
        Returns:
            Final status of all jobs
        """
        if self.manager_type == 'local':
            # Local jobs are already complete
            return {job_id: 'COMPLETED' for job_id in job_ids}
            
        total_jobs = len(job_ids)
        completed_jobs = set()
        failed_jobs = set()
        
        while len(completed_jobs) + len(failed_jobs) < total_jobs:
            # Get current status
            status = self.check_status(job_ids)
            
            # Update job counts
            new_completed = 0
            new_failed = 0
            for job_id, job_status in status.items():
                if job_status == 'COMPLETED' and job_id not in completed_jobs:
                    completed_jobs.add(job_id)
                    new_completed += 1
                elif job_status in ['FAILED', 'CANCELLED', 'TIMEOUT'] and job_id not in failed_jobs:
                    failed_jobs.add(job_id)
                    new_failed += 1
                    
            # Print progress
            if new_completed > 0 or new_failed > 0:
                self._logger.info(
                    f"Progress: {len(completed_jobs)}/{total_jobs} completed, "
                    f"{len(failed_jobs)}/{total_jobs} failed"
                )
                
            # Wait before next check
            if len(completed_jobs) + len(failed_jobs) < total_jobs:
                time.sleep(poll_interval)
                
        return status
        
            
    def check_status(self, job_ids: List[str]) -> Dict[str, str]:
        """
        Check status of submitted jobs
        
        Args:
            job_ids: List of job IDs to check
            
        Returns:
            Dictionary mapping job IDs to their status
        """
        if self.manager_type == 'local':
            # For local execution, jobs are already complete
            status = {}
            for job_id in job_ids:
                status[job_id] = 'COMPLETED'
            return status
        else:
            return self._manager.check_status(job_ids)


class SlurmJobManager(BaseJobManager):
    """Job manager for SLURM submission system"""
    
    DEFAULT_PARAMS = {
        'partition': 'normal',
        'nodes': 1,
        'ntasks': 1,
        'cpus_per_task': 8,
        'account': None
    }
    
    def __init__(self, config: Config):
        """
        Initialize SLURM job manager
        
        Args:
            config: Configuration with settings:
                slurm_params: SLURM submission parameters
                templates_dir: Directory with job templates
                max_jobs: Maximum concurrent jobs
                max_array_size: Maximum array job size
        """
        self.config = config or get_config()
        self._logger = setup_logger(__name__)
        
        # Set parameters
        self._slurm_params = self.DEFAULT_PARAMS.copy()
        self._slurm_params.update(config.get('job_manager.slurm_params', {}))
        
        # Set array job parameters
        self._max_array_size = config.get('max_array_size', 500)
        
    def create_job_script(self, job_spec: Dict[str, Any], template: str = None) -> str:
        """
        Create job submission script
        
        Args:
            job_spec: Job specifications
            template: Template name (not used in SLURM manager)
            
        Returns:
            Generated job script content
        """
        # Create basic SLURM script
        script = [
            "#!/bin/bash",
            f"#SBATCH --partition={self._slurm_params['partition']}",
            f"#SBATCH --nodes={self._slurm_params['nodes']}",
            f"#SBATCH --ntasks={self._slurm_params['ntasks']}",
            f"#SBATCH --cpus-per-task={self._slurm_params['cpus_per_task']}",
        ]
        
        if self._slurm_params['account']:
            script.append(f"#SBATCH --account={self._slurm_params['account']}")
            
        # Add job execution
        cmd = job_spec['command']
        if 'args' in job_spec:
            cmd = f"{cmd} {' '.join(map(str, job_spec['args']))}"
            
        script.extend([
            "",
            "# Change to job directory",
            f"cd {job_spec['job_path']}",
            "",
            "# Execute command",
            f"{cmd} > stdout.log 2> stderr.log"
        ])
        
        return "\n".join(script)
        
    def _create_array_script(self, job_specs: List[Dict[str, Any]], array_size: int) -> str:
        """
        Create SLURM array job script
        
        Args:
            job_specs: List of job specifications for array
            array_size: Size of job array
            
        Returns:
            Job script content
        """
        # Get common command and working directory
        base_cmd = job_specs[0]['command']
        
        # Create array script
        script = [
            "#!/bin/bash",
            f"#SBATCH --partition={self._slurm_params['partition']}",
            f"#SBATCH --nodes={self._slurm_params['nodes']}",
            f"#SBATCH --ntasks={self._slurm_params['ntasks']}",
            f"#SBATCH --cpus-per-task={self._slurm_params['cpus_per_task']}",
            f"#SBATCH --array=0-{array_size-1}%{self._max_array_size}",
            f"#SBATCH --job-name=array_job",
            f"#SBATCH --output=array_job_%A.out",
        ]
        
        if self._slurm_params['account']:
            script.append(f"#SBATCH --account={self._slurm_params['account']}")
            
        # Add job execution
        script.extend([
            "",
            "# Get job specs from array",
            "declare -a job_paths=(",
            *[f'"{spec["job_path"]}"' for spec in job_specs],
            ")",
            "",
            "declare -a job_args=(",
            *[f'"{" ".join(spec.get("args", []))}"' for spec in job_specs],
            ")",
            "",
            "# Change to job directory",
            "cd ${job_paths[$SLURM_ARRAY_TASK_ID]}",
            "",
            "# Execute command",
            f"{base_cmd} ${{job_args[$SLURM_ARRAY_TASK_ID]}} > stdout.log 2> stderr.log"
        ])
        
        return "\n".join(script)
        
    def submit_jobs(self, job_specs: List[Dict[str, Any]], batch_size: int = 500) -> List[str]:
        """
        Submit jobs to SLURM
        
        Args:
            job_specs: List of job specifications
            batch_size: Size of job batches for array jobs
            
        Returns:
            List of job IDs
        """
        job_ids = []
        
        # Split jobs into batches
        for i in range(0, len(job_specs), batch_size):
            batch = job_specs[i:i + batch_size]
            
            try:
                # Create temporary script file
                script_dir = Path(batch[0]['job_path']).parent
                script_file = script_dir / 'job_array.sh'
                
                # Write array script
                script_content = self._create_array_script(batch, len(batch))
                with open(script_file, 'w') as f:
                    f.write(script_content)
                
                # Submit array job
                result = subprocess.run(
                    ['sbatch', str(script_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Extract job ID from sbatch output
                # Output format: "Submitted batch job 123456"
                job_id = result.stdout.strip().split()[-1]
                job_ids.extend([f"{job_id}_{i}" for i in range(len(batch))])
                
                self._logger.info(f"Submitted array job {job_id} with {len(batch)} tasks")
                
            except subprocess.CalledProcessError as e:
                self._logger.error(f"Failed to submit array job: {e.stderr}")
                raise
                
        return job_ids
        
    def check_status(self, job_ids: List[str]) -> Dict[str, str]:
        """
        Check status of SLURM jobs
        
        Args:
            job_ids: List of job IDs
            
        Returns:
            Dictionary mapping job IDs to status
        """
        status = {}
        
        # Get unique array job IDs
        array_jobs = {job_id.split('_')[0] for job_id in job_ids}
        
        try:
            # Query job status using sacct
            result = subprocess.run(
                ['sacct', '-j', ','.join(array_jobs), '--format=JobID,State', '--parsable2'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse sacct output
            # Format: JobID|State
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            job_states = {}
            
            for line in lines:
                job_id, state = line.split('|')
                if '.batch' not in job_id:  # Skip batch job entries
                    array_id, task_id = job_id.split('_') if '_' in job_id else (job_id, None)
                    if task_id is not None:
                        job_states[f"{array_id}_{task_id}"] = state
                        
            # Map states to job IDs
            for job_id in job_ids:
                array_id, task_id = job_id.split('_')
                state = job_states.get(f"{array_id}_{task_id}", "UNKNOWN")
                
                # Map SLURM states to our status format
                if state in ['COMPLETED', 'COMPLETING']:
                    status[job_id] = 'COMPLETED'
                elif state in ['FAILED', 'TIMEOUT', 'CANCELLED']:
                    status[job_id] = state
                elif state in ['PENDING', 'RUNNING']:
                    status[job_id] = state
                else:
                    status[job_id] = 'UNKNOWN'
                    
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to check job status: {e.stderr}")
            status = {job_id: 'UNKNOWN' for job_id in job_ids}
            
        return status
        
    def get_results(self, job_ids: List[str]) -> Dict[str, Any]:
        """
        Get results from completed SLURM jobs
        
        Args:
            job_ids: List of job IDs
            
        Returns:
            Dictionary mapping job IDs to results
        """
        results = {}
        
        for job_id in job_ids:
            # Extract original job spec index
            array_id, task_id = job_id.split('_')
            task_id = int(task_id)
            
            # Get job path from original specs
            job_path = self._current_specs[task_id]['job_path']
            stderr_file = os.path.join(job_path, 'stderr.log')
            
            if not os.path.exists(stderr_file):
                results[job_id] = {'error': 'Job failed - no error log found'}
                continue
                
            with open(stderr_file, 'r') as f:
                error_content = f.read()
                if error_content.strip():
                    results[job_id] = {'error': error_content}
                else:
                    results[job_id] = {'status': 'completed'}
                    
        return results
