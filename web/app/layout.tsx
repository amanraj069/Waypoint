import type { Metadata } from "next";
import "./globals.css";
import { THEME_BOOTSTRAP } from "@/lib/theme";

export const metadata: Metadata = {
  title: "Waypoint · UAV mission studio",
  description:
    "Say a mission in plain English, watch an LLM decompose it, a deterministic "
    + "verifier check it, and a simulator fly it through a real place.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applies the stored theme before the first paint, so the page never
            flashes the wrong palette on the way in. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body className="h-full bg-bg text-text antialiased">{children}</body>
    </html>
  );
}
