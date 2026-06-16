# C8-2 FSNet+UDP Duplicate Audit

Decision: `NOT_DUPLICATE_RENDER_AND_ARCH_DIFFER`

- Current FullUDP checkpoint sha256: `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`
- FSNet+UDP checkpoint sha256: `25cc334f44c2fac979baad7f158526c9f8d751c21ea282974b0e4d9791fc0a27`
- ConvIR+UDP arch sha256: `038c349e191972c65fd95ff4b61ca7b97a9c075f1b1eda220ae6ed704faab543`
- FSNet+UDP arch sha256: `c0f0769d8b850f98fed8df6adbaef258cd67a48d24d187f8216560d027a84c3a`
- Architecture file identical: `False`
- Rendered output count: `600` train-derived images
- FullUDP-vs-FSNet output MAE mean/median: `0.00967903` / `0.00876259`
- FullUDP-vs-FSNet output PSNR mean/min: `38.9100` / `27.4357` dB
- Near-identical rendered outputs (MAE <= 1e-6 or PSNR >= 80): `0`
- Checkpoint-load note: `import-time FSNet_UDPNet fusion OCAB num_heads=1->2 because checkpoint bias tables are shape [1521,2]`

FSNet+UDP is not the same implementation/checkpoint/output as current FullUDP, so it is valid as a C8 conditional expert. Locked test remains untouched.
