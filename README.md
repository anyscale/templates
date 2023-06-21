# Anyscale Starter Templates

These templates are a set of minimal examples & tutorials for customers to run on the anyscale platform.

## Contributing Guide

If the template is generic to Ray & Ray libraries, please consider adding the template in: https://github.com/ray-project/ray/tree/master/doc/source/templates


To add a template:

1. Add your template as a directory under `templates/<your-template-name>`

    For example:

    ```text
    templates/my-awesome-template
        <name-of-your-template>/
            README.md
            <name-of-your-template>.ipynb/py
    ```

    Your template does not need to be a Jupyter notebook. It can also be presented as a
    Python script.

    All templates MUST have a `README.md` file.

2. Add your compute configuration under `configs/<your-template-name>` (for both AWS and GCE).

3. Update the product repo `backend/workspace-templates.yaml` to point to the new template added here after being merged.

4. Coming soon: update the integration tests in the product repo.