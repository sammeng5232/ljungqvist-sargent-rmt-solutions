# Solutions to Ljungqvist and Sargent, *Recursive Macroeconomic Theory* (4th edition, MIT Press, 2018)

Authors: Zijun Meng and Claude Opus 5

## Contents

Every exercise in the book has a worked solution: 254 exercises in 23 chapters.
Chapters 1, 4, 10, 12, 20, 25, 30 and 31 and the appendices have no exercises.
The exercise statements are not reproduced: read them in the book, alongside the
solutions. Each solution is headed by its exercise number. Notation, parameter values,
part labels such as (a) or (ii), and equation numbers such as (1) that belong to an
exercise are those of the exercise in the book.
Equations, sections, figures and programs cited as "(16.5.9)" or "Figure 11.9.4"
refer to Ljungqvist and Sargent (2018). The figures of these solutions are numbered
S*N*.*k* (chapter *N*) so that they cannot be confused with the book's figures.

| File | Contents |
|---|---|
| `Ljungqvist_Sargent_RMT_Solutions.tex` / `.pdf` | all the solutions in one document (239 pages), with a title page, a linked table of contents, and PDF bookmarks for every chapter and exercise |
| `figures/` | the 46 figures included by the document |
| `code/` | Python scripts for the numerical parts, one per chapter (see below) |

| Chapter | Exercises |
|---|---|
| 2. Time Series | 2.1 – 2.30 |
| 3. Dynamic Programming | 3.1 |
| 5. Linear Quadratic Dynamic Programming | 5.1 – 5.14 |
| 6. Search and Unemployment | 6.1 – 6.27 |
| 7. Recursive Competitive Equilibrium: I | 7.1 – 7.7 |
| 8. Equilibrium with Complete Markets | 8.1 – 8.28 |
| 9. Overlapping Generations | 9.1 – 9.13 |
| 11. Fiscal Policies in a Growth Model | 11.1 – 11.20 |
| 13. Asset Pricing Theory | 13.1 – 13.5 |
| 14. Asset Pricing Empirics | 14.1 – 14.24 |
| 15. Economic Growth | 15.1 – 15.6 |
| 16. Optimal Taxation with Commitment | 16.1 – 16.13 |
| 17. Self-Insurance | 17.1 – 17.5 |
| 18. Incomplete Markets Models | 18.1 – 18.7 |
| 19. Dynamic Stackelberg Problems | 19.1 – 19.3 |
| 21. Incentives and Insurance | 21.1 – 21.5 |
| 22. Equilibrium without Commitment | 22.1 – 22.5 |
| 23. Optimal Unemployment Insurance | 23.1 – 23.6 |
| 24. Credible Government Policies, I | 24.1 – 24.3 |
| 26. Two Topics in International Trade | 26.1 – 26.2 |
| 27. Fiscal-Monetary Theories of Inflation | 27.1 – 27.8 |
| 28. Credit and Currency | 28.1 – 28.11 |
| 29. Equilibrium Search, Matching, and Lotteries | 29.1 – 29.11 |

## Building the document

The `.tex` file is self-contained: the shared notation and the exercise environment
are defined in its preamble. It needs only the `figures/` folder next to it. Compile
from this directory with

```bash
pdflatex Ljungqvist_Sargent_RMT_Solutions.tex
```

Run it three times so that the contents page numbers and PDF bookmarks settle.
Microtype's font expansion is switched off in the preamble. With it on, pdfTeX
occasionally placed part of a line off the page.

## Code

The book's programs are in Matlab. The scripts in `code/` are Python replacements.
They need Python 3 with NumPy, SciPy and Matplotlib and no data files. Run a script
from the `code/` folder:

```bash
cd code
python ch21.py
```

Each script prints the numbers reported in the solutions and rewrites that chapter's
figures in `../figures/`. `rmtlib.py` holds the shared tools: the doubling algorithm
(`doublej`), linear-quadratic regulators (`olrp`, Howard improvement `lq_howard`), the
steady-state Kalman filter, and small helpers. Where the book reports results for
the same model, the scripts reproduce them first. Examples are Table 13.A.1, the
Section 19.5.3 Stackelberg example, Figure 23.2.1, and the Chapter 11 transition paths
(checked against the book's shooting algorithm).

| Script | Exercises | Run time |
|---|---|---|
| `ch2.py` | 2.1–2.30 (spectra, Kalman filter, permanent income, figures S2.1–S2.6) | 9 s |
| `ch5.py` | 5.1–5.14 (LQ regulators, Howard improvement, figure S5.1) | 2 s |
| `ch6.py` | 6.1–6.27 (McCall models, job-specific capital, Neal's model, figures S6.1–S6.2) | 1 min |
| `ch7.py` | 7.1–7.5 | 1 s |
| `ch8.py` | 8.1–8.28 (Arrow securities, Markov pricing, figure S8.1) | 3 s |
| `ch9.py` | 9.1–9.13 (overlapping generations, Laffer curves, figures S9.1–S9.2) | 3 s |
| `ch11.py` | 11.1–11.20 (perfect-foresight transitions by Newton's method, figures S11.1–S11.15) | 17 s |
| `ch13.py` | 13.1, 13.3, 13.5 and the Harrison–Kreps Table 13.A.1 | 1 s |
| `ch14.py` | 14.1–14.24 (Hansen–Jagannathan bounds, long-run risk, figures S14.1–S14.3) | 22 s |
| `ch15.py` | 15.3 (vintage capital, figure S15.1) | 3 s |
| `ch16.py` | 16.5, 16.11–16.13 (Lucas–Stokey economies, figure S16.1) | 4 s |
| `ch17.py` | 17.2, 17.5 | < 1 s |
| `ch18.py` | 18.1, 18.2, 18.3, 18.6 (Bertola mobility costs, Huggett and Bewley equilibria, figures S18.1–S18.2) | 30 s |
| `ch19.py` | 19.1–19.3 (Stackelberg and Markov perfect equilibria, figures S19.1–S19.2) | 7 s |
| `ch21.py` | 21.2–21.5 (moneylender, IMF, Phelan–Townsend linear programs, figures S21.1–S21.2) | 8 min |
| `ch22.py` | 22.3–22.5 (Kocherlakota, Kehoe–Levine, figures S22.1–S22.3) | 18 s |
| `ch23.py` | 23.1 (Hopenhayn–Nicolini: calibration and C(V) by policy iteration, figure S23.1) | 4 s |
| `ch24.py` | 24.1–24.3 (subgame-perfect values, figure S24.1) | 5 s |
| `ch26.py` | 26.1 (Bond–Park, figure S26.1) | 4 s |
| `ch27.py` | 27.1–27.8 (checks of the closed forms) | 1 s |
| `ch28.py` | 28.1–28.11 (checks of the closed forms) | 2 s |
| `ch29.py` | 29.1, 29.2, 29.4–29.6, 29.8 (island model, business-cycle search, European unemployment, figures S29.1–S29.2) | 19 s |

Chapter 3 has no script: its one exercise is analytical. Most of Chapters 7, 9, 13, 17,
27 and 28 is analytical too, and there the scripts only check the formulas numerically.

### Choices where the book leaves parameters open

- **16.11–16.13**: utility c + 0.2 log l (16.11), 0.8 log c + l on a 7-state chain
  (16.12), log c + 2.5 l (16.13); β = .95, b₀ = .2.
- **18.1, 18.3**: illustrative parameters. **18.6**: Huggett's (1993) calibration.
- **23.1**: Hopenhayn and Nicolini's calibration, as in the book.
- **28**: log utility and β = .9 in the numerical illustrations.
- **29.4–29.6**: a monthly matching calibration with 6% unemployment for h = 1 and
  b = .4. The turbulence model has a 40-year working life (see 29.6(d)).

## Apparent errors in the book

Each item is discussed where it arises in the solutions.

- **2.4(d)**: the formula for $E_t\sum .95^k y_{t+k}$ omits the constant term, which is needed unless μ = 0.
- **2.27**: the two branches of each stochastic discount factor are interchanged. Also, "Prob(s₁ = 0)" should read Prob(s_t = 0).
- **2.29**: the solution written as R Σ (A′)ʲ x_{t+j} holds only if R commutes with A′ (R belongs between (A′)ʲ and x_{t+j}). The second part is mislabelled "a".
- **5.7**: the lettering of the parts is slightly off (part (b) computes p₀^min, and the two inflation rates belong to parts (c) and (d)).
- **5.9**: the permanent-income condition is printed as βγ(1−δ) = 1. It should be β(γ + 1 − δ) = 1.
- **5.11**: with the stated parameters the perceived price converges to −900.
- **6.18**: the law of motion x_{t+1} = g(xφ) − δx should read x_{t+1} = x_t + g(x_tφ_t) − δx_t.
- **8.1**: "person 2 has no labor but is endowed with s units" should say person 1.
- **8.2**: "exercise 8.25" means exercise 8.1.
- **8.10(d)**: type 1's endowment is μ, not 1.
- **8.14**: booms should be y_t(1) = ȳ¹, not ȳ⁰.
- **11.7**: formula (11.10.16) is misprinted: it gives λ₂ = 3.53 at γ = 2 and λ₂ → ∞ as γ → ∞.
- **11.8**: the technology should read c_t + k_{t+1} = …, and a "where" clause is left incomplete.
- **11.10**: in R̄_{t+1}, f′(k_{t+1} − δ) should read f′(k_{t+1}) − δ.
- **11.18**: the Inada condition lim u′(c) as c ↓ 0 should be +∞, not 0.
- **11.20**: capital–labor ratio and consumption cannot stay exactly at their initial steady-state values before t = 10.
- **14.1**: the mean bill return 1.02 is probably 1.010 (Table 14.3.1, same data).
- **14.3**: the growth factor is written γ (the risk-aversion symbol) in places and ν elsewhere.
- **14.12**: the law of motion contains an undefined z_t.
- **14.13**: σ_μ and σ_z are missing from the parameter list.
- **14.19**: a missing matrix on x_t, the twisted mean (μ + φx_t − Cλ_t), and a missing transpose.
- **14.21**: part (c) refers to part (a) where part (b) is meant.
- **14.22(g)**: the recursion should be (3), not (2).
- **14.23**: the pricing kernel (1) has the wrong sign on the quadratic term.
- **14.24**: α_z(j) should be η_z(j), and u_{t+j} should be u_{t+1}.
- **16.7**: in the capital-tax revenue τ^a_t(r_{t+1} − δ)k_t the date index should be r_t.
- **16.11**: the conditions "H(0) > 1, H(1) > 0" are garbled. An interior leisure choice needs H′(0) > 1 > H′(1).
- **16.12**: the hint repeats the five-state chain of 16.11. Two random dates need seven states.
- **18.4**: the Bellman equation omits the non-capital income y.
- **19.3 / Section 19.5.3**: the large firm's rule is printed with the wrong sign. The printed vector is F in u_t = −F y_t.
- **21.3**: the utility function is written in w; it should be in ω.
- **21.4**: b_min is printed as 1 − y_max + .33. It should be a − y_max + .33.
- **22.5 / footnote 14**: the risk-free rate 1.0146 should be 1.0830 at c = .536.
- **24.2(l)**: the bound 10/19 requires randomization. With pure strategies the bound is .543689.
- **26.1 / (26.3.23)**: the denominator u_S(t_t^N) should be u_S(t_L^N).
- **27.5**: "income elasticity" should be "interest elasticity" (as in the title).
- **28.9**: the initial debts are reversed relative to what a stationary equilibrium from t = 0 requires. In (e)–(f) the interest rate reaches β⁻¹ at F = 1/(2(1+β)), not at β/(1+β).

## Not included

The book itself is not included in this repository, and neither are its exercise
statements. You need a copy of the book to read the exercises.
