import sys
import os
from gui.app import LandAnchorApp
from multiprocessing import freeze_support

# Constants and helpers should come after standard library imports
def get_resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(str(base_path), str(relative_path))

# Finally, your local application imports

if __name__ == "__main__":

    freeze_support()
    db_path = get_resource_path("landanchor.db")
    app = LandAnchorApp(db_path)
    app.mainloop()