# Constitution

*The guarantees this project keeps — across every version, regardless of implementation.*

> **This Constitution intentionally governs guarantees, not implementations.** Models, retrieval,
> ranking, and storage will change; what follows should not. That is why it reads differently from
> most project charters — it is closer to an interface contract than a manifesto.

## Purpose

Most tools that answer a question are built to tell you what is true. This one is built to
show you *how a conclusion was reached* — and to leave you free to disagree with it.

It exists on a simple premise: a conclusion you cannot inspect is a conclusion you cannot
trust. The worth of an answer is not its confidence but its accountability — the evidence
beneath it, the weighing that produced it, the conflicts it had to resolve, and the trail by
which it can be revisited. This project does not promise certainty. It promises a process,
and that the process is yours to examine.

## Guarantees

The project binds itself to the following. A *right* can be granted and revoked; a *guarantee* is a
commitment the project holds itself to — and one that you, or any contributor, may hold it to in turn:

1. **The evidence is inspectable** — every conclusion is shown with the sources it rests on; nothing
   is asserted that cannot be traced.
2. **Disagreement is surfaced** — where sources conflict, the conflict is shown, not smoothed away.
3. **The weighing is visible** — you can see *why* a conclusion currently prevails, not merely that it does.
4. **Every conclusion is contestable** — no verdict is final; each is offered with enough shown for you
   to challenge it.
5. **The process is reproducible** — the evidentiary path (the retrieved sources, the verification steps,
   and the recorded reasoning trace) can be re-derived from the same evidence, locally and without trusting
   a remote authority, even if the explanatory prose differs between runs.
6. **Change is traceable** — where revision tracking is enabled, a shift in evidence or consensus stays
   visible; conclusions have a history, not only a present.

These guarantees do not depend on any particular implementation. They are what the software is *for*.

## Responsibilities

Those who build on this project agree to uphold those guarantees. A contribution earns its place
to the degree that it makes evidence *more* inspectable, surfaces disagreement rather than
hiding it, shows its reasoning instead of asking for trust, and leaves a trace others can
follow.

The governing question for any change is one sentence: **does it strengthen or weaken the
guarantees above?** Cleverness that weakens them is not an improvement.

## Boundaries

This project intentionally does **not** claim to tell you what is true, to be a final
authority, or to be correct because it is confident. It can be wrong. The point was never that
it does not err — it is that when it does, the path to seeing and correcting the error stays
open. Its legitimacy is the fairness and inspectability of its process, not a presumption of
its own infallibility. *Every conclusion stays challengeable.*

## Evolution

Implementations will change. Models, retrieval, ranking, the means of cross-checking — all may
be replaced, and should be, as better methods appear. **These guarantees should not.** They are
the part of the project meant to stay recognizable across every version that follows. When
little of today's code survives, this page should still describe what the project is.

When the choice arises — and it will — between preserving today's implementation and preserving today's
guarantees, preserve the guarantees. The implementation exists to serve them, not the other way around.

## Amendments

The guarantees change rarely, in public, with the rationale recorded here — never merely because a
maintainer preferred it. (A formal article on the amendment process itself is deliberately left for later,
once there is enough experience to write it well.)

- **2026-06-26 — G5 and G6 clarified for falsifiability.** A review asking *"could a stranger catch us if we
  failed this guarantee?"* found both true but underspecified: G5 did not define what "reproduce" means (the
  evidentiary *path*, not byte-identical prose), and G6 did not state its scope (revision tracking enabled).
  Both were tightened so the claim is testable. No capability changed.
