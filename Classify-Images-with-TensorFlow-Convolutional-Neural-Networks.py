!pip install google-cloud-aiplatform
!pip install --user tensorflow-text
!pip install --user tensorflow-datasets
!pip install protobuf==3.20.1

import os

if not os.getenv("IS_TESTING"):
    # Automatically restart kernel after installs
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
