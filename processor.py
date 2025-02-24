# processor.py

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
import job_scraper

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<h1>Job Matcher API</h1>")
        elif self.path == "/download-jobs":
            # Serve the CSV file for download
            try:
                with open("all_jobs.csv", "rb") as file:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/csv')
                    self.send_header('Content-Disposition', 'attachment; filename="all_jobs.csv"')
                    self.end_headers()
                    self.wfile.write(file.read())
            except FileNotFoundError:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "CSV file not found"}).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not Found"}).encode())

    def do_POST(self):
        try:
            status = 200
            response = {}

            if self.path == '/process-resume':
                print("\n[POST] /process-resume")
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )
                file_item = form['resume']

                with open('uploaded_resume.pdf', 'wb') as f:
                    f.write(file_item.file.read())

                try:
                    data = job_scraper.process_resume('uploaded_resume.pdf')
                    response = {'skills': data.get('skills', [])}
                except Exception as e:
                    print("Error processing resume:", e)
                    response = {'error': f'Resume parsing failed: {str(e)}'}
                    status = 500

            elif self.path == '/find-jobs':
                print("\n[POST] /find-jobs")
                content_length = int(self.headers['Content-Length'])
                post_data = json.loads(self.rfile.read(content_length))
                
                try:
                    roles = post_data.get('roles', [])
                    skills = post_data.get('skills', [])
                    filters = post_data.get('filters', {})

                    print("  Received roles:", roles)
                    print("  Received skills:", skills)
                    print("  Received filters:", filters)

                    results = job_scraper.full_processing_flow(
                        resume_path='uploaded_resume.pdf',
                        roles=roles,
                        skills=skills,
                        filters=filters
                    )

                    response = {
                        'topMatches': results['top_matches'],
                        'totalJobs': results.get('total_jobs_found', 0)
                    }
                except Exception as e:
                    print("Error finding jobs:", e)
                    response = {'error': f'Job search failed: {str(e)}'}
                    status = 500

            else:
                response = {'error': 'Invalid endpoint'}
                status = 404

            self.send_response(status)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            print("Uncaught error in do_POST:", e)
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    print("Starting server at http://localhost:8000")
    server = HTTPServer(('localhost', 8000), RequestHandler)
    server.serve_forever()

if __name__ == '__main__':
    run_server()
