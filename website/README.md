# EVA Website

A New End-to-end Framework for Evaluating Voice Agents (EVA)

Built with React, TypeScript, Vite, and Tailwind CSS.

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd website
npm install
```

### Development

Run the development server with hot module replacement:

```bash
npm run dev
```

The site will be available at http://localhost:5173

### Production Build

Build for production:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

### Linting

```bash
npm run lint
```

## Project Structure

```
src/
├── components/
│   ├── architecture/    # System architecture diagram
│   ├── conversation/    # Interactive conversation demo
│   ├── hero/            # Landing page hero section
│   ├── layout/          # Navbar, Section wrapper
│   ├── leaderboard/     # Results tables, charts, and visualizations
│   ├── limitations/     # Known limitations section
│   ├── metrics/         # Metrics explorer and judge prompt viewer
│   └── acknowledgements/
├── data/                # Static data for demos and leaderboard
└── styles/              # Theme configuration
```

The site uses hash-based routing with tabs: intro, architecture, metrics, leaderboard-oss, demo, limitations, and acknowledgements.
