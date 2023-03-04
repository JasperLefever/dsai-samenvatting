# Module 2. Univariate statistics

## Central Dendency and Dispersion

### Mesures of central tendency

- **Arithmetic Mean**:

  - Gemiddelde
  - Som van alle waardes gedeeld door het aantal waardes
  - $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$

- **Median**:

  - Middenste waarde -> mediaan (uit **gesorteerde** lijst)
  - Als er een even aantal waardes is, dan is de mediaan het gemiddelde van de twee middenste waardes

- **Mode**:
  - Meest voorkomende waarde
  - Als er meerdere waardes zijn die even vaak voorkomen, dan is er geen mode

### Mesures of dispersion

- **Range**:

  - Verschil tussen de grootste en kleinste waarde
  - $\text{Range} = \text{max} - \text{min}$

- **Quartiles**:

  - Deelverzameling van de waardes
  - Deelverzamelingen zijn gesorteerd
  - Deelverzamelingen zijn even groot
  - Er zijn 4 delen
  - Deelverzamelingen zijn als volgt genoemd:
    - 1e kwartiel: 25%
    - 2e kwartiel: 50%
    - 3e kwartiel: 75%
  - $\text{Q}_1 = \text{median}(\text{min}, \text{median})$
  - $\text{Q}_2 = \text{median}(\text{min}, \text{max})$
  - $\text{Q}_3 = \text{median}(\text{median}, \text{max})$

- **Interquartile Range**:

  - Verschil tussen de 3e en 1e kwartiel
  - $\text{IQR} = \text{Q}_3 - \text{Q}_1$

- **Variance**:

  - Gemiddelde van de kwadraten van de afwijkingen van de waardes ten opzichte van het gemiddelde
  - Voor bij sample:
    - $\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
  - voor bij populatie:
    - $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$

- **Standard Deviation**:
  - Wortel van de variantie
  - $\sigma = \sqrt{\sigma^2}$

### Summary

| Mesurement Level | Center               | Spread Distribution                                            |
| ---------------- | -------------------- | -------------------------------------------------------------- |
| Qualitative      | Mode                 | -                                                              |
| Quantitative     | Average, Mean median | Variance, Standard Deviation Median Range, Interquartile Range |

Summary of symbols:
|/|Population| Sample|
|------|-------|------|
| number of elements | $N$ | $n$ |
| average or mean | $\mu$ | $\bar{x}$ |
| variance | $\sigma^2 = \frac{1}{N}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$ |
| standard deviation | $\sigma$ | $s$ |

---

## Data visualization

- Chart type overview
  | Mesurement Level | Chart Type |
  | ---------------- | ---------- |
  | Qualitative | Bar chart |
  | Quantitative | Histogram, Boxplot, Density plot |

### Simple graphs

- don't use pie charts
  - comparing angles is difficult

### Interpretation of graphs

- **Tips**

  - Label the axes
  - Use a title
  - Name the units
  - Add label that clarifies the meaning of the graph

- **Data distortion**
  - Misleading graphs
