import type { NextConfig } from "next";

/**
 * The API runs as a separate process (uvicorn, port 8000 by default) because it
 * is the Python pipeline (planner, verifier, simulator), not a Node service.
 * Proxying it under /api keeps the browser on one origin, so there is no CORS
 * configuration to get wrong and no API base URL to bake into a build.
 */
const API = process.env.WAYPOINT_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Next gzips everything it serves, proxied responses included, and a gzip
  // stream is flushed in blocks: the mission's server-sent events sat in the
  // compressor and the page never left "queued". Nothing here is big enough to
  // be worth compressing on localhost.
  compress: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
