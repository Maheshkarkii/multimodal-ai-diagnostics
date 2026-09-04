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
      <body className="min-h-screen flex flex-col bg-industrial-50 text-industrial-900">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="bg-white border-t border-industrial-200 py-6 text-center text-xs text-industrial-500">
          AI Field Engineer — Diagnostic & Troubleshooting System (Phase 1 to Phase 12 Integrated)
        </footer>
      </body>
    </html>
  );
}
