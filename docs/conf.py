"""Sphinx configuration for cfspopcon."""

from inspect import Parameter, signature

from sphinx.ext.autodoc import ClassDocumenter, FunctionDocumenter
from sphinx.ext.autodoc.importer import get_class_members
from sphinx.ext.intersphinx import missing_reference
from sphinx.util.inspect import stringify_signature

import cfspopcon
from cfspopcon.algorithm_class import Algorithm

project = "cfspopcon"
copyright = "2023, Commonwealth Fusion Systems"
author = cfspopcon.__author__
version = cfspopcon.__version__
release = version

# -- General configuration

# warn for missing references. Can be quite loud but at least we then have
# ensured that all links to other classes etc work.
nitpicky = True

add_module_names = False

# note that github actions seems to be on openssl 3.0.
# if you aren't locally, that can cause different behaviour.
linkcheck_ignore = [
    # server is incompatible with openssl 3.0 default, see e.g.
    # https://github.com/urllib3/urllib3/issues/2653
    r"https://doi.org/10.2172/7297293",
    r"https://doi.org/10.2172/1372790",
    # these work but linkcheck doesn't like them..
    r"https://doi.org/10.2172/1334107",
    r"https://doi.org/10.13182/FST91-A29553",
    r"https://doi.org/10.1080/10420150.2018.1462361",
    r"https://library.psfc.mit.edu/catalog/online_pubs/MFE_formulary_2014.pdf",
    r"https://doi.org/10.13182/FST11-A11650",
    r"https://github.com/cfs-energy/cfspopcon/blob/main/docs/doc_sources/getting_started.ipynb",
    r"https://doi.org/10.13182/FST43-67",
    r"https://www.tandfonline.com/doi/full/10.13182/FST43-67",
    r"https://www-internal.psfc.mit.edu/research/alcator/data/fst_cmod.pdf",
    # these links in the time_independent_inductances_and_fluxes notebook are on private servers that are sometimes down
    r"https://fire.pppl.gov/iaea06_ftp7_5_matsukawa.pdf",
    r"https://escholarship.org/content/qt78k0v04v/qt78k0v04v_noSplash_c44c701847deffab65024dd9ceff9c59.pdf?t=p15pc5",
    r"https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=585f5eb3f62f3bd76f3d667c1df357562f54c084",
    r"https://fire.pppl.gov/Snowmass_BP/FIRE.pdf",
    r"https://www.ipp.mpg.de/16208/einfuehrung",
    r"https://www.ipp.mpg.de/16701/jet",
    r"https://iopscience.iop.org/article/10.1088/1009-0630/13/1/01",
    r"https://www-internal.psfc.mit.edu/research/alcator/data/fst_cmod.pdf",
    # These bib resources fail due to "403 Client Error: Forbidden for url"
    r"https://doi.org/10.1103/PhysRevLett.121.055001",
    r"https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.055001",
]
linkcheck_retries = 5
linkcheck_timeout = 120
linkcheck_report_timeouts_as_broken = False

source_suffix = ".rst"

# If docs change signficantly such that navigation depth is more, this setting
# might need to be increased
html_theme_options = {
    "navigation_depth": 3,
}

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
highlight_language = "python3"

# -- Options for HTML output

html_static_path = ["static"]
html_theme = "sphinx_rtd_theme"
html_domain_indices = False
html_use_index = False
html_show_sphinx = False
htmlhelp_basename = "cfspopconDoc"
python_maximum_signature_line_length = 90
#
# -- extensions and their options
#

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # linkcode to point to github would be nicer
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "nbsphinx",
]

nitpick_ignore = [("py:class", "Ellipsis")]

# -- nbsphinx
exclude_patterns = ["_build", "static"]
nbsphinx_execute = "never"

# -- autodoc
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}
autoclass_content = "both"
autodoc_typehints = "signature"

# -- doctest
doctest_global_setup = """
from cfspopcon import *
"""

# -- intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pint": ("https://pint.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- matplotlib plot_directive
# only plot a png
plot_formats = ["png"]
# don't show link to the png
plot_html_show_formats = False

# -- copybutton
# make copy paste of code blocks nice on copy remove the leading >>> or ... of
# code blocks and remove printed output. Their default is usually good but
# currently a bit broken so we need the below
copybutton_exclude = ".lineos"
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# -- bibtex bibliography
bibtex_bibfiles = ["refs.bib"]


# register a resolve function to help sphinx with the resolve references sphinx
# couldn't note: sphinx doesn't stop calling listeners once one with lower
# priority has returned a good result so this function is called pretty much
# for every cross-reference, thus we need to filter out the cases we actually
# want to handle
def resolve(app, env, node, contnode):
    """Custom reference resolver."""
    ret_node = None

    if node["refdomain"] == "py" and node["reftype"] == "class":
        py = env.domains["py"]

        # type hint links are transformed into something like
        # :py:class:`numpy.float64` but `numpy.float64` is actually documented
        # as a :py:attr:. We just use the general :py:obj: here which should be
        # fine as long as there aren't any name collisions in numpy
        if "numpy" in node["reftarget"]:
            node["reftype"] = "obj"
            ret_node = missing_reference(app, env, node, contnode)

        # This is a similar fix to above. We have cases where we use a generic
        # type e.g. cfspopcon.strict_base.T and that is is a :py:attr: so we
        # run into the same case as above. Same sledgehammer approach of
        # just using :py:obj: for any missing links at this tag
        # The additional list of type hints is here because they don't get resolved to
        # e.g. cfspopcon.algorithm_class.GenericFunctionType, but just the plain type
        elif "cfspopcon" in node["reftarget"] or node["reftarget"] in [
            "LabelledReturnFunctionType",
            "GenericFunctionType",
            "Params",
            "Ret",
            "FunctionType",
        ]:
            node["reftype"] = "obj"
            ret_node = py.resolve_xref(env, node["refdoc"], app.builder, node["reftype"], node["reftarget"], node, contnode)

        # patch Self return types to point to the class the function is defined
        # on
        elif "typing_ext" in node["reftarget"]:
            node["reftarget"] = node["py:class"]
            ret_node = py.resolve_xref(env, node["refdoc"], app.builder, node["reftype"], node["reftarget"], node, contnode)

        elif "pint" in node["reftarget"]:
            s = node["reftarget"]
            if s.startswith("pint") and s.endswith("Quantity"):
                node["reftarget"] = "pint.Quantity"
                ret_node = missing_reference(app, env, node, contnode)

        elif node["reftarget"] == "matplotlib.pyplot.Axes":
            node["reftarget"] = "matplotlib.axes.Axes"
            ret_node = missing_reference(app, env, node, contnode)

    return ret_node


# the below workaround is adopted from:
# https://github.com/celery/celery/blob/1683008881717d2f8391264cb2b6177d85ff5ea8/celery/contrib/sphinx.py#L42
# which is BSD3 licensed see:
# https://github.com/celery/celery/blob/1683008881717d2f8391264cb2b6177d85ff5ea8/LICENSE#L1


# wraps_ufunc returns a class which leads to sphinx ignoring the function
# This is a custom documenter to ensure automodule correctly lists wrapped functions
# and creates a better signature for them. Setting the signature object on the actual class (like for Algorithms)
# isn't possible because the __call__ function is always a member of the class and setting it on an instance
# does not work.
class FunctionWrapperDocumenter(FunctionDocumenter):
    """Document a wraps_ufunc wrapped function"""

    # this means `autowraps_ufunc` is a new autodoc directive
    objtype = "wraps_ufunc"
    # but document those as functions
    directivetype = "function"
    member_order = 11

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return super().can_document_member(member, membername, isattr, parent) and hasattr(member, "unitless_func")

    def format_args(self):
        fw = self.object
        sig = signature(fw)
        return stringify_signature(sig, unqualified_typehints=True)

    def document_members(self, all_members=False):
        super(FunctionDocumenter, self).document_members(all_members)

    def get_object_members(self, want_all: bool):
        members = get_class_members(self.object, self.objpath, self.get_attr, self.config.autodoc_inherit_docstrings)
        unitless_func = members.get("unitless_func", None)
        if unitless_func is not None:
            unitless_func.object.__doc__ = "A scalar and not unit aware version of the above function."
            # the unitless function will get documented as a member of the FuncitionWrapper class
            # but sphinx pops the first argument because it thinks that's the "self" so we monkey patch around that
            # by prepending a parameter that gets thrown away
            tmp_param = Parameter("tmp", kind=Parameter.POSITIONAL_ONLY)
            s = signature(unitless_func.object)
            new_sig = s.replace(parameters=[tmp_param, *s.parameters.values()], return_annotation=s.return_annotation)
            unitless_func.object.__signature__ = new_sig
        return False, [unitless_func]


class AlgDocumenter(ClassDocumenter):
    """Document a Algorithm instance."""

    objtype = "popcon_alg"
    # data so that we don't get the "class" prefix in sphinx
    directivetype = "data"
    member_order = 21

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, Algorithm)

    def add_directive_header(self, sig: str) -> None:
        super(ClassDocumenter, self).add_directive_header(sig)

    def get_object_members(self, want_all: bool):
        members = get_class_members(self.object, self.objpath, self.get_attr, self.config.autodoc_inherit_docstrings)
        return False, [m for k, m in members.items() if k in {"run", "update_dataset", "return_keys"}]

    def format_signature(self, **kwargs) -> str:
        return ""

    def get_doc(self):
        return None

    def import_object(self, raiseerror: bool = False) -> bool:
        ret = super().import_object(raiseerror)
        self.doc_as_attr = False
        return ret


def setup(app):
    # default is 900, intersphinx is 500
    app.connect("missing-reference", resolve, 1000)
    app.add_css_file("theme_overrides.css")
    app.add_autodocumenter(FunctionWrapperDocumenter)
    app.add_autodocumenter(AlgDocumenter)
