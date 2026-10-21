#!/bin/sh
# Keep this Mac awake with the lid closed so the live bot keeps trading.
#   sudo ./keepawake.sh on       # stay awake with the lid shut (plug in power!)
#   sudo ./keepawake.sh off      # back to normal sleep
#   ./keepawake.sh status
# On a laptop, closing the lid sleeps the Mac whatever is running; 'pmset disablesleep' is the only switch
# that stops that. While it's on, don't put the closed Mac in a bag: it keeps running and warming up.
case "$1" in
  on)  pmset -a disablesleep 1 && echo "sleep disabled: the Mac stays awake with the lid closed. Keep it plugged in." ;;
  off) pmset -a disablesleep 0 && echo "sleep enabled again: closing the lid will sleep the Mac (and pause the bot)." ;;
  status|"")
       if pmset -g | grep -q "SleepDisabled.*1"; then echo "lid-closed awake: ON"; else echo "lid-closed awake: OFF (closing the lid sleeps the Mac)"; fi
       pmset -g batt | sed -n 1,2p ;;
  *)   echo "usage: sudo $0 on|off   or   $0 status"; exit 1 ;;
esac
