"""Prompts for generalized two-image change detection."""

SYSTEM_INSTRUCTION = (
    "You are a visual analysis assistant. Compare two images of the same scene captured at "
    "different times. Image A is the newer view; Image B is the older view. "
    "Your primary task is to identify meaningful changes between the images and name the "
    "objects involved. Be precise, objective, and conservative when uncertain."
)

PROMPT_PAIR_DIRECT = r"""
Two images are provided in order:
- First image = A (NEW / later)
- Second image = B (OLD / earlier)

Goal:
Identify real visual changes between A and B and return the name of each affected object.

Guidelines:
- Focus on persistent objects and scene elements (furniture, fixtures, structures, equipment, signage, vegetation, vehicles, etc.).
- Ignore transient elements unless they are the main change (e.g., a person or vehicle that appears only in one image).
- Do not report differences that are likely caused only by lighting, shadows, exposure, blur, reflection, occlusion, or small camera viewpoint shift.
- If the two images do not show substantially the same scene, mark them as not comparable.
- Use clear, specific object names (e.g., "office chair", "floor lamp", "cardboard box", "window blind").

Workflow:
1. Describe what is visible in A and B independently.
2. Decide whether the images are comparable (same scene with sufficient overlap).
3. List candidate changes: objects added, removed, or visibly modified.
4. Keep only changes supported by direct visual evidence in both images.
5. For each accepted change, provide the object name and a short explanation.

Change types:
- added: object appears in A but not in B
- removed: object appears in B but not in A
- modified: same object is present in both images but its appearance, position, quantity, or state clearly differs

Return ONLY valid JSON with exactly these keys:

{
  "comparable": boolean,
  "overlap_score": number,
  "viewpoint_shift": "none" | "small" | "moderate" | "large",
  "image_a_description": string,
  "image_b_description": string,
  "changes": [
    {
      "change_type": "added" | "removed" | "modified",
      "object_name": string,
      "object_names_alternatives": [string],
      "location": string,
      "description": string,
      "before_state": string,
      "after_state": string,
      "confidence": number
    }
  ],
  "no_change": boolean
}

Field rules:
- object_name: primary noun phrase naming the changed object (no verbs, no full sentences)
- object_names_alternatives: 1-3 close synonyms or alternative names for the same object
- location: brief spatial description of where the object appears
- description: one concise sentence describing what changed
- before_state: how the object appears in B
- after_state: how the object appears in A
- confidence: float from 0.0 to 1.0

If comparable=true but no reliable changes remain:
- no_change=true
- changes=[]

If comparable=false:
- no_change=true
- changes=[]
"""
