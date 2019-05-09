from src.tagging_tool import tagging_tool
from tkinter import *

def main():
    root = Tk()
    app = tagging_tool(master=root)
    root.mainloop()


if __name__ == "__main__":
	main()
