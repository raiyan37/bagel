import './styles.css';

interface StatRowProps {
  label: string;
  value: number | string;
  unit?: string;
  highlight?: boolean;
}

export function StatRow({ label, value, unit, highlight = false }: StatRowProps) {
  return (
    <div className={`stat-row ${highlight ? 'stat-row-highlight' : ''}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </span>
    </div>
  );
}
