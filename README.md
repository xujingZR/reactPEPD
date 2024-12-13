# README

# reactPEPD

---

reactPEPD is a metastable phase prediction method that combines thermodynamic theory with dynamic factors. The core concept of reactPEPD is **“consistency”** in crystalline structure evolution, which is reflected in two key aspects:

1. **Relative contents of elements:** The elemental composition during synthesis changes gradually without abrupt shifts, exhibiting a trend that aligns with the target composition.
2. **Similarity of intermediate structures:** The system evolves along a trajectory where intermediate structures remain similar to the target product, minimizing the likelihood of drastic structural changes.

Building on these considerations, we developed reactPEPD to facilitate predictions of metastable phases qualitatively! 🎉👏

The related paper is currently in preparation—hope you’ll enjoy it once it’s ready! 😊

---

### Usage

Now, many functions and results that this method can provided are still under development. 😔

Please stay tuned! I’ll working hard to complete it in the upcoming months~ 👏 👏

```python
from reactPEPD import reactPEPD

# for example: Li-Co-O
reactPEPD(
		L = "Li",
		M = "Co",
		A = "O",
		L_source = "Li2CO3",
		M_source = "Co3O4",
		api_key = "your-api-key",
		root_dir = "your-work-path"
		)
```

Unfortunately, the current code is limited to specific cases, such as close-packed structures and using carbonate as the Li precursor. But don’t worry! This limitation is simply due to the current implementation in the reactPEPD package and does not reflect the method’s overall capability. If needed, you can modify the code to suit the specific situations you’re interested in—feel free to adapt it as you like! 😊