import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/layout/Navbar";

export const metadata: Metadata = {
  title: "AI Field Engineer — Multimodal Autonomous Diagnostics",
  description: "Industrial autonomous diagnostic reasoning across vision, acoustics, sensor telemetry, and OEM technical documentation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-[#f5f5f7] text-[#1d1d1f]">
        <Navbar />
        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
          {children}
        </main>
        <footer className="bg-white/80 backdrop-blur-md border-t border-black/[0.06] py-8 text-center text-xs text-[#86868b]">
          <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>AI Field Engineer — Multimodal Autonomous Diagnostics & Troubleshooting</span>
            <span>Industrial Reliability Platform</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
