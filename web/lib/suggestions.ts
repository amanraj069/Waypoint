/**
 * Something to type, when the box is empty and the place is new.
 *
 * These are written per world because a good first mission names things that
 * are actually there — the point of the whole exercise is that "photograph
 * Bara Gumbad from two sides" lands on a building with a position and a height,
 * and a generic "survey the area" teaches you nothing about that.
 */

export const SUGGESTIONS: Record<string, string[]> = {
  delhi: [
    "climb over the Secretariat, photograph Rashtrapati Bhavan from the east, then come home",
    "fly the length of Kartavya Path below the treeline, then circle India Gate twice",
    "photograph the North Block and the South Block from above Vijay Chowk, then land",
    "survey Sansad Bhavan from four sides at a safe stand-off, then return to the lawns",
  ],
  lodhi: [
    "photograph Bara Gumbad from two sides, fly north up the ashoka avenue, then come home",
    "make a low pass over the lake, then photograph Sheesh Gumbad from the south",
    "inspect Safdarjung's Tomb from three angles without leaving the garden",
    "fly the avenue at treetop height, then cross the old grove and land",
  ],
  chowk: [
    "fly the length of Chandni Chowk below the rooftops, from the Town Hall to the Lahori Gate",
    "photograph the Jama Masjid minarets from the east, then come back down the street",
    "climb above the roofs, photograph Gurudwara Sis Ganj Sahib, then descend into the street",
    "run west to Fatehpuri Masjid at 14 m, then return over the rooftops",
  ],
  munnar: [
    "climb the Pothamedu ridge, photograph the Mattupetty dam, then come back over the tea estate",
    "photograph Anamudi from a safe stand-off, then make a low pass over Mattupetty Lake",
    "survey the Kanan Devan tea estate at a constant height above the ground",
    "fly to Echo Point and photograph the lake from above it, then return to town",
  ],
};

/**
 * A fallback for worlds nobody wrote suggestions for — an OSM import, say.
 * Landmark names are the only thing that is certainly true about a place.
 */
export function suggestionsFor(worldId: string | null, landmarks: string[]): string[] {
  if (worldId && SUGGESTIONS[worldId]) return SUGGESTIONS[worldId];
  if (!landmarks.length) return ["survey the area 100 m north of base, then return and land"];
  const [first, second, third] = landmarks;
  const lines = [`photograph ${first} from two sides, then come home`];
  if (second) lines.push(`fly to ${second}, hold, then return to base and land`);
  if (third) lines.push(`survey ${first} and ${third}, keeping clear of everything else`);
  return lines;
}
