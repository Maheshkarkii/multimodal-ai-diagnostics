/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        apple: {
          bg: '#f5f5f7',
          canvas: '#fbfbfd',
          card: '#ffffff',
          'card-hover': '#fafafc',
          subtle: '#f5f5f7',
          muted: '#e8e8ed',
          border: '#d2d2d7',
          'border-light': 'rgba(0, 0, 0, 0.06)',
          'border-subtle': 'rgba(0, 0, 0, 0.04)',
          text: {
            primary: '#1d1d1f',
            secondary: '#6e6e73',
            tertiary: '#86868b',
            quaternary: '#a1a1a6',
          },
          blue: '#0071e3',
          'blue-hover': '#0077ed',
          dark: '#1d1d1f',
          black: '#000000',
        },
        industrial: {
          50: '#f5f5f7',
          100: '#e8e8ed',
          200: '#d2d2d7',
          300: '#b0b0b8',
          400: '#86868b',
          500: '#6e6e73',
          600: '#515154',
          700: '#333336',
          800: '#1d1d1f',
          900: '#111113',
          950: '#000000',
        },
        brand: {
          blue: '#0071e3',
          amber: '#f5a623',
          emerald: '#34c759',
          rose: '#ff3b30',
        }
      },
      borderRadius: {
        'apple-sm': '10px',
        'apple-md': '14px',
        'apple-lg': '18px',
        'apple-xl': '22px',
        'apple-2xl': '28px',
      },
      boxShadow: {
        'apple-xs': '0 1px 3px rgba(0, 0, 0, 0.04)',
        'apple-sm': '0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02)',
        'apple-card': '0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)',
        'apple-hover': '0 12px 32px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.04)',
        'apple-modal': '0 24px 48px rgba(0, 0, 0, 0.12), 0 4px 12px rgba(0, 0, 0, 0.06)',
      }
    },
  },
  plugins: [],
}
