""" pyTA version, required package versions, and utilities for checking."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
#          Bertrand Thirion <bertrand.thirion@inria.fr>
# License: new BSD

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
from distutils.version import LooseVersion


__version__ = '0.0.1'
_PYTA_INSTALL_MSG = "See {0} for installation information.".format(
                    'https://github.com/CherkaouiHamza/pyta')


# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').
REQUIRED_MODULE_METADATA = (
    ('numpy', {
        'min_version': '1.14.0',
        'required_at_installation': True,
        'install_info': _PYTA_INSTALL_MSG}),
    ('scipy', {
        'min_version': '1.0.0',
        'required_at_installation': True,
        'install_info': _PYTA_INSTALL_MSG}),
    ('joblib', {
        'min_version': '0.11',
        'required_at_installation': True,
        'install_info': _PYTA_INSTALL_MSG}),
    # XXX need to find a fix for the 'scikit-learn' / 'sklearn' problem
    # ('scikit-learn', {
    #     'min_version': '0.19.2',
    #     'required_at_installation': True,
    #     'install_info': _HEMOLEARN_INSTALL_MSG}),
    ('prox_tv', {
        # version not propperly set by the maintainer
        'min_version': '0.0.0',
        'required_at_installation': True,
        'install_info': _PYTA_INSTALL_MSG}),
    # ('carpet', {
    #     # version not propperly set by the maintainer
    #     'min_version': '0.0.0',
    #     'required_at_installation': True,
    #     'install_info': _PYTA_INSTALL_MSG}),
    ('matplotlib', {
        'min_version': '2.1.2',
        'required_at_installation': False}),
    )


def _import_module_with_version_check(
        module_name,
        minimum_version,
        install_info=None):
    """Check that module is installed with a recent enough version
    """
    try:
        module = __import__(module_name)
    except ImportError as exc:
        user_friendly_info = ('Module "{0}" could not be found. {1}').format(
            module_name,
            install_info or 'Please install it properly to use pyta.')
        exc.args += (user_friendly_info,)
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (
            'A {module_name} version of at least {minimum_version} '
            'is required to use pyta. {module_version} was found. '
            'Please upgrade {module_name}').format(
                module_name=module_name,
                minimum_version=minimum_version,
                module_version=module_version)

        raise ImportError(message)

    return module


def _check_module_dependencies(is_pyta_installing=False):
    """Throw an exception if pyta dependencies are not installed.

    Parameters
    ----------
    is_pyta_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.

    Throws
    -------
    ImportError
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_pyta_installing and
                not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            if 'import_name' in module_metadata.keys():
                module_name = module_metadata['import_name']
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
                install_info=module_metadata.get('install_info'))
