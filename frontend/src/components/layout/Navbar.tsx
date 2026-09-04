"use client";

import Link from "next/link";
import { Wrench, Activity, BookOpen, ShieldCheck } from "lucide-react";

export function Navbar() {
  return (
    <header className="bg-industrial-900 border-b border-industrial-800 text-white sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Link href="/" className="flex items-center space-x-2">
            <div className="p-2 bg-brand-blue rounded-lg text-white">
              <Wrench className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight">AI Field Engineer</span>
              <span className="ml-2 text-xs bg-industrial-800 text-industrial-300 px-2 py-0.5 rounded border border-industrial-700">
                v1.0.0
              </span>
            </div>
          </Link>
        </div>

        <nav className="flex items-center space-x-6 text-sm font-medium">
          <Link
            href="/"
            className="text-industrial-300 hover:text-white transition flex items-center space-x-1"
          >
            <Activity className="w-4 h-4" />
            <span>Dashboard</span>
          </Link>
          <Link
            href="/cases/new"
            className="bg-brand-blue hover:bg-sky-600 text-white px-4 py-2 rounded-lg font-semibold transition flex items-center space-x-2 shadow-sm"
          >
            <Wrench className="w-4 h-4" />
            <span>New Case</span>
          </Link>
        </nav>
      </div>
    </header>
  );
}
