# CHURRO — product vision (not current engineering scope)

This is background for product direction, not a spec. Nothing here should
be implemented unless the roadmap in `CLAUDE.md` has explicitly reached it.
Read this when the conversation is about product direction, the UI phase,
or long-term design — not during normal engineering work on the current
roadmap step.

## Important product principle

CHURRO is not being built merely to compete with Cursor or Claude Code
feature-for-feature. The goal is to explore a different interaction model
for AI. The core engineering foundation comes first; the interface and
experimental interaction model come later.

## The UI phase (future — do not start yet)

The entire underlying CHURRO engine should be substantially complete
before UI implementation begins. Once the core engineering roadmap
(see `CLAUDE.md`) is far enough along, the product interface becomes its
own dedicated phase, covering:

1. Home page
2. First-prompt experience
3. Main coding/workspace interface
4. Files sidebar
5. AI activity/interaction view
6. Model switching UX
7. Flowchart-like relationships between code, tasks, agents, models, and
   files
8. Interactive AI workspace
9. Polished, considered visual design
10. Animations and UX refinement

The UI must NOT simply imitate Cursor, Claude Code, or VS Code — it should
express CHURRO's own interaction model, not clone an existing one.

## Long-term experience vision (brainstorming, not requirements)

Eventually CHURRO should offer a genuinely new way of interacting with AI.
These are exploratory ideas for that future, not commitments or specs:

- AI represented as interactive entities
- Visual task graphs
- Code ↔ AI relationships
- AI ↔ AI relationships
- Direct manipulation of AI processes
- Spatial workspaces
- Human + AI collaboration
- AI handoffs
- Task branching
- Multiple simultaneous AI roles
- Cross-domain workspaces

None of this should influence engineering decisions on the current
roadmap. It exists so that future UI/product planning has the original
thinking to work from, rather than reinventing it.
