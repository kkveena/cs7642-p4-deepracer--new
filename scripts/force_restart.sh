#!/bin/bash
echo "🛑 STOPPING ALL DeepRacer Processes..."

# Kill Python clients first
pkill -u $USER -9 python || true

# Kill Simulation Engine
pkill -u $USER -9 deepracer || true
pkill -u $USER -9 apptainer || true
pkill -u $USER -9 gzserver || true
pkill -u $USER -9 rosmaster || true

# Kill Port Hog (The TA's main concern)
echo "🧹 Clearing Port 9194..."
/usr/sbin/fuser -k -n tcp 9194 > /dev/null 2>&1 || true

# Clear Shared Memory (The "Zombie" killer)
rm /dev/shm/*$USER* 2>/dev/null || true

# Clean up temporary lock files
rm -f /tmp/ros* 2>/dev/null

echo "✅ System Cleaned. You can now launch training."
