import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lankara — NeuroX 1.0",
  description: "Autonomous B2B retail agent with human-in-the-loop approval",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[radial-gradient(ellipse_at_top,_#0f2918_0%,_#020617_55%)] text-slate-100 antialiased">
        <header className="border-b border-slate-800/80 bg-slate-950/40 backdrop-blur">
          <nav className="mx-auto flex max-w-4xl items-center gap-6 px-6 py-4 text-sm">
            <a href="/" className="text-base font-semibold tracking-tight text-emerald-400">
              Lankara
            </a>
            <a href="/" className="text-slate-400 hover:text-white">
              Run
            </a>
            <a href="/queue" className="text-slate-400 hover:text-white">
              Approval Queue
            </a>
          </nav>
        </header>
        <main className="mx-auto max-w-4xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
