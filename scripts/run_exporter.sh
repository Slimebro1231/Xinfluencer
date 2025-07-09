#!/bin/bash
echo "🚀 Starting node_exporter..."
cd /home/ubuntu/node_exporter
nohup ./node_exporter > /home/ubuntu/node_exporter/node_exporter.log 2>&1 &
echo "✅ node_exporter started in the background." 