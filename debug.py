def main():
    global window
    import tkinter as tk
    greeting = tk.Label(text="Whats going on in this dang camera")
    window = tk.Tk()
    greeting.pack()
    input()

def update(skew, distance, midpoint, edges, webcam_frame):
    global window
    pass