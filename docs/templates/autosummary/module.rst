{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
