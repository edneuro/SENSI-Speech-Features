import platform
from utils.system_info import gather_system_info


def test_system_info_returns_dict_with_expected_keys():
    info = gather_system_info()
    assert set(info.keys()) >= {'os', 'python_version', 'gpu', 'system_ram_gb'}


def test_system_info_os_matches_platform():
    info = gather_system_info()
    assert platform.system() in info['os']


def test_system_info_python_version_is_string():
    info = gather_system_info()
    assert isinstance(info['python_version'], str)
    assert info['python_version'].count('.') >= 2  # e.g. "3.10.12"


def test_system_info_gpu_has_name_and_vram_keys():
    info = gather_system_info()
    assert 'name' in info['gpu']
    assert 'vram_gb' in info['gpu']
