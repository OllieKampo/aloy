import curses
import random
import time

# create a curses object
stdscr = curses.initscr()

# get the window size
height, width = stdscr.getmaxyx()

# define two color pairs, 1- header/footer , 2 - dynamic text, 3 - background
curses.start_color()
curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLUE)

# Write a header and footer, first write colored strip, then write text
stdscr.bkgd(curses.color_pair(3))
stdscr.addstr(0, 0, " " * width, curses.color_pair(1))
stdscr.addstr(height - 1, 0, " " * (width - 1), curses.color_pair(1))
stdscr.addstr(0, 0, " Curses Dynamic Text Example", curses.color_pair(1))
stdscr.addstr(height - 1, 0, " Type q - to quit ", curses.color_pair(1))
stdscr.addstr(3, 5, "SIMULATED SENSOR VALUES", curses.A_BOLD)
stdscr.refresh()

# Cycle to update text. Enter a 'q' to quit
k = 0
stdscr.nodelay(1)
while (k != ord('q')):
    # write 10 lines text with a label and then some random numbers
    for i in range(1, 11):
        stdscr.addstr(4 + i, 5, "Sensor " + str(i) + " : ", curses.A_BOLD)
        stdscr.addstr(4 + i, 20, str(random.randint(10, 99)), curses.color_pair(2))
        time.sleep(2)
        k = stdscr.getch()

curses.endwin()
