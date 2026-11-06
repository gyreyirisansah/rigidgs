RigidGS License
================

The **University of Wyoming** holds all the ownership rights on the
*Software* named **RigidGS**.

The *Software* was developed by Paul Sansah Gyreyiri and Dr. Diksha Shukla of
the Secure Sensing and Learning Research Lab at the University of Wyoming,
and is still being developed by the *Licensor*.

*Licensor*'s goal is to allow the research community to use, test and
evaluate the *Software*.

## 1.  Definitions

*Licensee* means any person or entity that uses the *Software* and
distributes its *Work*.

*Licensor* means the owner of the *Software*, i.e. the University of
Wyoming.

*Software* means the original work of authorship made available under this
License, i.e. RigidGS, excluding the third-party code described in Section 6
below.

*Work* means the *Software* and any additions to or derivative works of the
*Software* that are made available under this License.

## 2.  Purpose

This license is intended to define the rights granted to the *Licensee* by
the Licensors under the *Software*.

## 3.  Rights granted

For the above reasons Licensors have decided to distribute the *Software*.
Licensors grant non-exclusive rights to use the *Software* for research
purposes to research users (both academic and industrial), free of charge,
without right to sublicense. The *Software* may be used "non-commercially",
i.e., for research and/or evaluation purposes only.

Subject to the terms and conditions of this License, you are granted a
non-exclusive, royalty-free, license to reproduce, prepare derivative works
of, publicly display, publicly perform and distribute its *Work* and any
resulting derivative works in any form.

## 4.  Limitations

**4.1 Redistribution.** You may reproduce or distribute the *Work* only if (a) you do
so under this License, (b) you include a complete copy of this License with
your distribution, and (c) you retain without modification any copyright,
patent, trademark, or attribution notices that are present in the *Work*.

**4.2 Derivative Works.** You may specify that additional or different terms apply
to the use, reproduction, and distribution of your derivative works of the *Work*
("Your Terms") only if (a) Your Terms provide that the use limitation in
Section 2 applies to your derivative works, and (b) you identify the specific
derivative works that are subject to Your Terms. Notwithstanding Your Terms,
this License (including the redistribution requirements in Section 4.1) will
continue to apply to the *Work* itself.

**4.3** Any other use without prior consent of Licensors is prohibited. Research
users explicitly acknowledge having received from Licensors all information
allowing them to appreciate the adequacy between the *Software* and their
needs and to undertake all necessary precautions for its execution and use.

**4.4** The *Software* is provided as source code. In case of using the
*Software* for a publication or other results obtained through the use of
the *Software*, users are strongly encouraged to cite the corresponding
publication as explained in the documentation of the *Software*.

## 5.  Disclaimer

THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL PURPOSES
WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS. YOU MUST CONTACT THE LICENSORS
FOR ANY UNAUTHORIZED USE: pgyreyir@uwyo.edu OR dshukla@uwyo.edu . ANY SUCH
ACTION WILL CONSTITUTE A FORGERY. THIS *SOFTWARE* IS PROVIDED "AS IS" WITHOUT
ANY WARRANTIES OF ANY NATURE AND ANY EXPRESS OR IMPLIED WARRANTIES, WITH
REGARDS TO COMMERCIAL USE, PROFESSIONAL USE, LEGAL OR NOT, OR OTHER, OR
COMMERCIALISATION OR ADAPTATION. UNLESS EXPLICITLY PROVIDED BY LAW, IN NO
EVENT, SHALL THE LICENSORS OR THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA, OR
PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*SOFTWARE* OR THE USE OR OTHER DEALINGS IN THE *SOFTWARE*.

## 6.  Files subject to other licenses

RigidGS builds on and vendors several third-party projects under `external/`.
Each retains its own original license, found in that project's own directory;
none of them are relicensed by this document:

- `external/gaussian-splatting/` -- the original 3D Gaussian Splatting code
  (Inria / MPII), under the Gaussian-Splatting License
  (`external/gaussian-splatting/LICENSE.md`).
- `external/sam2/` -- SAM 2 (Meta), under its own license
  (`external/sam2/LICENSE`, `external/sam2/LICENSE_cctorch`).
- `external/lama/` -- LaMa, under its own license (`external/lama/LICENSE`).
- `external/Depth-Anything-V2/` -- Depth Anything V2, under its own license
  (`external/Depth-Anything-V2/LICENSE`). Note that while the base code is
  Apache-licensed, the larger pretrained checkpoints distributed by that
  project are released under CC BY-NC 4.0, a separate, more restrictive term
  that this document does not alter.

In addition, `external_overrides/gaussian-splatting/` and
`eval/render_fg_bg.py` are modified copies of the original Inria/MPII
Gaussian-Splatting code rather than original RigidGS authorship, and remain
subject to the Gaussian-Splatting License referenced above, not to this
document.
