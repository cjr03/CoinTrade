#!/bin/sh
# Run the live bot as a background service (launchd): starts at login, restarts within 30 s if it crashes,
# and holds a no-idle-sleep assertion while it runs. Lid-closed operation also needs: sudo ./keepawake.sh on
#   ./service.sh install | uninstall | status | logs
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"
LABEL=com.cointrade.bot
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
case "$1" in
  install)
    if pgrep -f "btcbot.app" >/dev/null && ! launchctl list | grep -q "$LABEL"; then
      echo "a bot is already running outside the service; stop it first (pkill -f btcbot.app), then install"; exit 1
    fi
    mkdir -p "$HOME/Library/LaunchAgents" state
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-is</string><string>$DIR/run_live.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$DIR/state/live_app.out</string>
  <key>StandardErrorPath</key><string>$DIR/state/live_app.out</string>
</dict></plist>
PL
    launchctl bootstrap "gui/$(id -u)" "$PLIST" && echo "installed: the bot now runs as $LABEL" ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null; rm -f "$PLIST"; echo "removed $LABEL (the bot is stopped)" ;;
  status)
    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state =|pid =|last exit" || echo "service not installed"
    ./keepawake.sh status ;;
  logs) tail -40 state/live_app.out ;;
  *) echo "usage: $0 install|uninstall|status|logs"; exit 1 ;;
esac
