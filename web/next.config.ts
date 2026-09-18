import type { NextConfig } from "next";

/**
 * The API runs as a separate process (uvicorn, port 8000 by default) because it
 * is the Python pipeline — planner, verifier, simulator — not a Node service.
 * Proxying it under /api keeps the browser on one origin, so there is no CORS
 * configuration to get wrong and no API base URL to bake into a build.
 */
const API = process.env.WAYPOINT_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
