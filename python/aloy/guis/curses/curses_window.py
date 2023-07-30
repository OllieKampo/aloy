import curses
import pyfiglet

title = pyfiglet.figlet_format("Sensor Reading Example", font="small")
stdscr = curses.initscr()

curses.start_color()
curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

# Write the title
stdscr.addstr(1, 0, title, curses.color_pair(1))
stdscr.refresh()

# Write the two sensor readings
win1 = curses.newwin(9, 44, 6, 4)
win1.addstr(8, 0, "Sensor 1 Reading", curses.A_BOLD)
win2 = curses.newwin(9, 44, 6, 50)
win2.addstr(8, 0, "Sensor 2 Reading", curses.A_BOLD)

# Write the sensor values
value1 = pyfiglet.figlet_format("21", font="doom")
win1.addstr(0, 0, value1, curses.color_pair(2))
value2 = pyfiglet.figlet_format("38", font="doom")
win2.addstr(0, 0, value2, curses.color_pair(2))

# Refresh the windows
win1.refresh()
win2.refresh()

# Wait for a key press to exit
stdscr.getch()
curses.endwin()
