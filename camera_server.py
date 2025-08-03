# Legacy imports removed; this wrapper simply forwards to the new app.

from singularity_vision.app import main

# This file is kept for backward-compatibility with existing tooling.
# It simply forwards execution to the restructured application entry-point.

if __name__ == "__main__":
    main() 