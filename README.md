# ComfyUI Lycoris Implementation V1
___

* This implementation is built to specifically address the LYCORIS modules not correctly loading into the ComfyUI structure.
* Each node is designed to handle the specific functionality of the LYCORIS modules.
* Precomputation is crucial for the functionality of these nodes, as the additive nature of the sampler system in ComfyUI demands linear and specific rules.
* The nodes are designed to be used in a specific order to ensure that the precomputation is done correctly.

* Tested:
* * LOHA - Functionally loads and works as expected.
* * Locon - Functionally loads and works as expected.
* * Lokr - Semi tested, but requires further testing.
* * Boft - untested, but likely requires precomputation mixed with runtime computation against the target model.
* * Oft - untested, likely requires precomputation mixed with runtime computation against the target model.
* * glora - untested, unlikely to need runtime computation.
* * blora - untested, uncertain if it requires runtime computation.

# Usage
___
* LoadLycorisNode: This node is used to load the LYCORIS modules into the ComfyUI structure.
* * This node is built specifically to handle precomputation of the LYCORIS modules within a cached structure.

* PrecomputeSaveLycorisNode: This node is built to handle runtime distillation and saving of lycoris into a precomputed form.
* * Warning, they can get very bloated and very large after precomputation.

* Precomputed lycoris are essentially treated as standard loras. They do not require additional traits and can be loaded at runtime, however their size varies greatly from their current form.

# Installing
___
* Until a package is created you can clone the repo and move it into your custom_nodes directory.

# Limitations
___
* Tested primarily with SDXL on ComfyUI
* The nodes are designed to work with the LYCORIS modules and may not function correctly with other modules.
* The default LORA fallback may malfunction or malform your lora during precomputation, so it is recommended to use standard lora loader if you get malformed responses from standard loras.

# Licensing
___
Apache License 2.0 is included in the repository.