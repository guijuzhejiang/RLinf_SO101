# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Register SO101 robot to be compatible with more ManiSkill tasks.

This module automatically adds SO101 support to tasks that support multiple robots.
"""

import mani_skill.envs  # Ensure all envs are registered
from mani_skill.utils.registration import REGISTERED_ENVS

# Import SO101 agent to ensure it's registered
from rlinf.envs.maniskill.agents.so101 import SO101


def register_so101_to_tasks():
    """
    Register SO101 robot to all compatible ManiSkill tasks.
    
    This function adds 'so101' to the SUPPORTED_ROBOTS list of tasks
    that already support similar robots (like so100, panda, fetch, etc.)
    """
    
    # Tasks that we want to add SO101 support to
    # These are tasks that already support so100 or similar arm robots
    compatible_tasks = []
    
    for env_id, env_spec in REGISTERED_ENVS.items():
        try:
            env_cls = env_spec.cls
            if hasattr(env_cls, 'SUPPORTED_ROBOTS') and env_cls.SUPPORTED_ROBOTS:
                robots = env_cls.SUPPORTED_ROBOTS
                # Check if this task supports arm-like robots (so100, panda, etc.)
                if any(r in ['so100', 'panda', 'fetch', 'xarm6_robotiq', 'widowxai'] 
                       for r in robots if isinstance(r, str)):
                    if 'so101' not in robots:
                        compatible_tasks.append((env_id, env_cls))
        except Exception:
            pass
    
    # Add SO101 to compatible tasks
    added_tasks = []
    for env_id, env_cls in compatible_tasks:
        try:
            if isinstance(env_cls.SUPPORTED_ROBOTS, list):
                env_cls.SUPPORTED_ROBOTS.append('so101')
                added_tasks.append(env_id)
            elif isinstance(env_cls.SUPPORTED_ROBOTS, tuple):
                env_cls.SUPPORTED_ROBOTS = list(env_cls.SUPPORTED_ROBOTS) + ['so101']
                added_tasks.append(env_id)
        except Exception:
            pass
    
    return added_tasks


def get_so101_compatible_tasks():
    """
    Get list of all tasks that SO101 is compatible with.
    """
    compatible = []
    for env_id, env_spec in REGISTERED_ENVS.items():
        try:
            env_cls = env_spec.cls
            if hasattr(env_cls, 'SUPPORTED_ROBOTS') and env_cls.SUPPORTED_ROBOTS:
                if 'so101' in env_cls.SUPPORTED_ROBOTS:
                    compatible.append(env_id)
        except Exception:
            pass
    return sorted(compatible)


# Auto-register when this module is imported
_registered_tasks = register_so101_to_tasks()

if _registered_tasks:
    print(f"[SO101] Registered SO101 robot to {len(_registered_tasks)} additional tasks")


# List of all tasks SO101 is now compatible with
SO101_COMPATIBLE_TASKS = get_so101_compatible_tasks()

