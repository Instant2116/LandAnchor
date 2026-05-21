import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.app import LandAnchorApp

if __name__ == "__main__":
    
    app = LandAnchorApp("landanchor.db")
    app.mainloop()