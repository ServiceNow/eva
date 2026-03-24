import type { ReactNode } from 'react';
import { motion } from 'framer-motion';

interface SectionProps {
  id: string;
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  wide?: boolean;
}

export function Section({ id, title, subtitle, children, className = '', wide = false }: SectionProps) {
  return (
    <section id={id} className={`pt-28 pb-20 px-4 sm:px-6 lg:px-8 ${className}`}>
      <div className={`${wide ? 'max-w-[1600px]' : 'max-w-7xl'} mx-auto`}>
        {title && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-3">{title}</h2>
            {subtitle && <p className="text-lg text-text-secondary max-w-3xl mx-auto">{subtitle}</p>}
          </motion.div>
        )}
        {children}
      </div>
    </section>
  );
}
