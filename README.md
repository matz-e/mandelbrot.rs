# Mandelbrot.rs

A simplistic Mandelbrot implementation in Rust, to compare with the ISPC
benchmarks (and practice some SIMD).

Current runtimes of the Rust version, on a Intel® Core™ i3-6100:
```
❯ cargo +nightly run --release
    Finished release [optimized] target(s) in 0.03s
     Running `target/release/mandelbrot`
Time of serial run:                            349.730 megacycles
Time of serial run:                            346.377 megacycles
Time of serial run:                            347.106 megacycles
Wrote 'mandelbrot-serial.ppm'
Time of complex run:                           359.835 megacycles
Time of complex run:                           359.050 megacycles
Time of complex run:                           358.558 megacycles
Wrote 'mandelbrot-complex.ppm'
Time of simd run:                               53.078 megacycles
Time of simd run:                               53.242 megacycles
Time of simd run:                               52.124 megacycles
Wrote 'mandelbrot-simd.ppm'
```

Compared to runtimes with ISPC:
```
❯ ../ispc/build/bin/mandelbrot
@time of ISPC run:                      [50.563] million cycles
@time of ISPC run:                      [50.106] million cycles
@time of ISPC run:                      [50.063] million cycles
[mandelbrot ispc]:              [50.063] million cycles
Wrote image file mandelbrot-ispc.ppm
@time of serial run:                    [368.979] million cycles
@time of serial run:                    [352.912] million cycles
@time of serial run:                    [349.603] million cycles
[mandelbrot serial]:            [349.603] million cycles
Wrote image file mandelbrot-serial.ppm
                                (6.98x speedup from ISPC)
```
