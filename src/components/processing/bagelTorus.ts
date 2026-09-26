/* A bagel is a torus, so this is the classic ASCII donut: points on the torus
   are rotated, projected, z-buffered into a character grid and shaded by how
   much each surface normal faces the light. */

const W = 44;
const H = 22;
const SHADES = '.,-~:;=!*#$@';

/* Torus radii and viewer distance, in the same arbitrary units as the projection. */
const R_TUBE = 1;
const R_HOLE = 2;
const DEPTH = 5;
const SCALE = (W * DEPTH * 3) / (8 * (R_TUBE + R_HOLE));

/** One frame of a torus lit from the upper left, z-buffered into a character grid. */
export function renderBagel(spinX: number, spinZ: number): string {
  const cells = new Array<string>(W * H).fill(' ');
  const depths = new Array<number>(W * H).fill(0);
  const cosX = Math.cos(spinX);
  const sinX = Math.sin(spinX);
  const cosZ = Math.cos(spinZ);
  const sinZ = Math.sin(spinZ);

  for (let theta = 0; theta < Math.PI * 2; theta += 0.09) {
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    const ringX = R_HOLE + R_TUBE * ct;
    const ringY = R_TUBE * st;

    for (let phi = 0; phi < Math.PI * 2; phi += 0.03) {
      const cp = Math.cos(phi);
      const sp = Math.sin(phi);

      const x = ringX * (cosZ * cp + sinX * sinZ * sp) - ringY * cosX * sinZ;
      const y = ringX * (sinZ * cp - sinX * cosZ * sp) + ringY * cosX * cosZ;
      const z = DEPTH + cosX * ringX * sp + ringY * sinX;
      const inverseZ = 1 / z;

      // Characters are about twice as tall as wide, so vertical scale is halved.
      const px = Math.round(W / 2 + SCALE * inverseZ * x);
      const py = Math.round(H / 2 - (SCALE / 2) * inverseZ * y);
      if (px < 0 || px >= W || py < 0 || py >= H) continue;

      const luminance =
        cp * ct * sinZ - cosX * ct * sp - sinX * st + cosZ * (cosX * st - ct * sinX * sp);
      if (luminance <= 0) continue;

      const cell = px + py * W;
      if (inverseZ <= depths[cell]) continue;
      depths[cell] = inverseZ;
      cells[cell] = SHADES[Math.min(SHADES.length - 1, Math.floor(luminance * 8))];
    }
  }

  const rows: string[] = [];
  for (let row = 0; row < H; row += 1) rows.push(cells.slice(row * W, row * W + W).join(''));
  return rows.join('\n');
}
