# Stripped-down version of the screenshot script, to work with textractor and auto-advance things to rip large quantities of text

import pyautogui
import time

delay_time = 5
num_lines_to_take = 500

print(f"Begin {time.time()}")
for s in range(delay_time):
    print(f"Get the game ready! Ripping in {delay_time - s}")
    time.sleep(1)

# Take and save a certain number of screenshots
for scrsht in range(num_lines_to_take):
    pyautogui.click()
    print("click")
    time.sleep(0.05)