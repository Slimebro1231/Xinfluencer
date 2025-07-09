from flask import Flask, jsonify, render_template_string
import requests
import logging
from database.manager import get_recent_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIGURATION ---
NODE_EXPORTER_URL = "http://157.10.162.127:9100/metrics"

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Xinfluencer AI Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; color: #1c1e21; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: auto; }
        h1 { color: #0056b3; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .panel { background-color: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); padding: 20px; }
        h2 { border-bottom: 2px solid #0056b3; padding-bottom: 10px; margin-top: 0; }
        pre { background-color: #e9ecef; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #0056b3; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .approved-true { color: green; font-weight: bold; }
        .approved-false { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Xinfluencer AI Dashboard</h1>
        
        <div class="grid">
            <div class="panel">
                <h2>System Status</h2>
                <pre id="system-status">Loading...</pre>
            </div>

            <div class="panel">
                <h2>GPU Metrics (from H200)</h2>
                <pre id="gpu-metrics">Loading...</pre>
            </div>
        </div>

        <div class="panel" style="margin-top: 20px;">
            <h2>Recent Pipeline Results</h2>
            <table id="results-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Query</th>
                        <th>Response</th>
                        <th>Approved</th>
                        <th>Review Score</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Data will be populated here -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-status').textContent = `Pipeline Status: ${data.status}\\nLast Check: ${new Date().toLocaleTimeString()}`;
                });
        }

        function updateMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('gpu-metrics').textContent = `Error: ${data.error}`;
                    } else {
                        // A simple display of some key GPU metrics
                        let metricsText = 'Metrics not available';
                        if (data.metrics) {
                           metricsText = 'Sample subset of GPU metrics:\\n\\n';
                           const interestingMetrics = [
                               'nvidia_gpu_duty_cycle',
                               'nvidia_gpu_memory_used_bytes',
                               'nvidia_gpu_power_draw_watts',
                               'nvidia_gpu_temperature_celsius'
                           ];
                           interestingMetrics.forEach(key => {
                               if(data.metrics[key]) {
                                   metricsText += `${key}: ${data.metrics[key]}\\n`;
                               }
                           });
                        }
                        document.getElementById('gpu-metrics').textContent = metricsText;
                    }
                });
        }

        function updateResults() {
            fetch('/api/results')
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.querySelector('#results-table tbody');
                    tableBody.innerHTML = '';
                    data.results.forEach(result => {
                        const row = document.createElement('tr');
                        row.innerHTML = \`
                            <td>${new Date(result.timestamp).toLocaleString()}</td>
                            <td>${result.query}</td>
                            <td>${result.response}</td>
                            <td class="approved-${result.approved.toString()}">${result.approved}</td>
                            <td>${result.review_score.toFixed(2)}</td>
                        \`;
                        tableBody.appendChild(row);
                    });
                });
        }

        function refreshData() {
            updateStatus();
            updateMetrics();
            updateResults();
        }

        setInterval(refreshData, 5000); // Refresh every 5 seconds
        document.addEventListener('DOMContentLoaded', refreshData);
    </script>
</body>
</html>
"""

# --- API Endpoints ---
@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """Return the current status of the pipeline."""
    # This is a placeholder; a more advanced implementation could
    # check if the main service process is running.
    return jsonify({"status": "Running"})

@app.route('/api/results')
def api_results():
    """Return the latest results from the database."""
    results = get_recent_results(limit=20)
    return jsonify({"results": results})

@app.route('/api/metrics')
def api_metrics():
    """Fetch and return metrics from the node_exporter."""
    try:
        response = requests.get(NODE_EXPORTER_URL, timeout=5)
        response.raise_for_status()
        
        # Extremely basic parsing for demonstration
        metrics = {}
        for line in response.text.split('\\n'):
            if line.startswith('#') or line == '':
                continue
            parts = line.split(' ')
            key = parts[0]
            value = parts[-1]
            metrics[key] = value

        return jsonify({"metrics": metrics})
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not fetch metrics from node_exporter: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True) 