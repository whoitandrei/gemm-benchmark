# gemm-benchmark

Внешний бенчмарк GEMM: замеры чужих библиотек, с которыми сравниваются Vulkan-ядра Adept. Сами Vulkan-ядра
Adept меряются в Adept, программой `benchmarks/op_bench`: так они работают через настоящий пайплайн библиотеки.
Этот бенчмарк меряет ориентиры тем же способом и пишет CSV в том же формате, поэтому результаты обеих программ
можно сложить в одну таблицу и один график. План работ — [docs/roadmap.md](docs/roadmap.md).

## Структура

```
src/
  device.hpp/.cpp      название процессора и число потоков
  problem.hpp/.cpp     задача GEMM: пакет, M, K, N, типы входа и накопления, общая B; генерация входов
  half.hpp             f16: перевод float <-> half с округлением к ближайшему чётному
  reference.hpp/.cpp   эталон (double) и проверка точности результата
  kernel.hpp           интерфейс ядра: prepare() -> run() ... sync() -> readC(); таймер GPU на блок вызовов
  measure.hpp/.cpp     замер как в Adept (benchmark::measure): прогрев, раунды по N вызовов, медианы
  kernels/
    registry.cpp       список всех ядер сборки
    ggml_cpu.cpp       ggml, CPU-бэкенд (ggml_mul_mat)
    xnnpack.cpp        XNNPACK, fully connected f32
    ggml_vulkan.cpp    ggml, Vulkan-бэкенд (ggml_mul_mat), только f16
    cublas.cpp         cuBLAS (cublasGemmStridedBatchedEx), только NVIDIA: ядра cublas и cublas_tc
bench/
  main.cpp             gemm_bench: опции как у op_bench, проверка точности, замер, отчёт
  cases.hpp/.cpp       список случаев: имена, формы, типы
  report.hpp/.cpp      строки RESULT и CSV в формате op_bench
tests/
  test_gemm.cpp        эталон, f16, ядра после вызовов в очереди, gpu_ms, замер (Catch2)
scripts/
  plot.py              GOP/s по CSV gemm_bench и op_bench
  peaks.csv            пики устройств с источниками, описание в docs/devices.md
  gpu_counters.py      счётчики GPU M1 из трассы Instruments, по каждому compute pipeline
```

## Сборка

```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

Ядра подключаются опциями CMake. По умолчанию собирается только `ggml_cpu` (`GEMM_WITH_GGML=ON`) и `cublas`,
если CMake нашёл CUDA Toolkit. Если не включено ни одно ядро, CMake останавливается с ошибкой `no kernels`.

| Опция | Ядра | По умолчанию |
|---|---|---|
| `GEMM_WITH_GGML` | `ggml_cpu` | `ON` |
| `GEMM_WITH_XNNPACK` | `xnnpack` | `OFF`, сборка несколько минут |
| `GEMM_WITH_GGML_VULKAN` | `ggml_vulkan` | `OFF`, нужен Vulkan SDK, сборка долгая |
| `GEMM_WITH_CUBLAS` | `cublas`, `cublas_tc` | `ON`, если найден CUDA Toolkit 11.4+ |

Все библиотеки, кроме CUDA, скачиваются при первой настройке CMake.

### ggml и XNNPACK

```bash
cmake -S . -B build -DGEMM_WITH_XNNPACK=ON
```

- `ggml_cpu` — ggml v0.26.0, только CPU-бэкенд, умножение через `ggml_mul_mat` с llamafile sgemm,
  как в llama.cpp. Свой пул потоков, без OpenMP.
- `xnnpack` — fully connected оператор f32. Пакет с общей B считается одним оператором,
  иначе — отдельный оператор на каждую матрицу пакета.
- Обе считают B весами: упаковка B (у ggml — переупаковка в N x K) идёт в `prepare()` и не замеряется,
  как в инференсе.
- Только f32, пакет и общая B поддерживаются.
- Потоков столько, сколько ядер у машины (`cpuThreads()`, то есть `std::thread::hardware_concurrency`). На процессорах с
  энергоэффективными ядрами (Apple M, Intel 12+) это не всегда быстрее: ggml делит работу между потоками
  поровну, и потоки на медленных ядрах тормозят весь запуск.

### ggml-vulkan

Vulkan-бэкенд той же ggml v0.26.0, что у `ggml_cpu`, — GEMM из llama.cpp. Зрелая открытая реализация на
том же API, что Adept. На M1 это единственный внешний GPU-ориентир.

```bash
cmake -S . -B build -DGEMM_WITH_GGML_VULKAN=ON
```

- Нужен Vulkan SDK с `glslc` и SPIRV-Headers. На macOS SDK работает через MoltenVK; если SDK поставлен не
  системно, перед cmake и запуском: `source ~/VulkanSDK/<версия>/setup-env.sh`. ggml скачивается, сборка
  долгая: ggml генерирует и компилирует несколько сотен шейдеров.
- Умножение — `ggml_mul_mat`, один граф на весь пакет, общая B — broadcast ggml. B считается весами, как
  у `ggml_cpu`: перестановка в N x K идёт в `prepare()`. Транспонированная A не поддерживается.
- Только f16-входы. `f16/f32` — `GGML_PREC_F32`. `f16/f16` — `GGML_PREC_DEFAULT`: ggml копит в f16, если
  устройство это умеет, иначе сам переходит на f32. Результат у ggml всегда f32, для накопления в f16 он
  переводится в f16 в `readC()`, вне замера.
- f32 не поддерживается: на GPU с f16 ggml-vulkan кладёт f32-входы в shared memory как f16, и точность
  не проходит допуск f32.
- Вызов — `ggml_backend_graph_compute_async`, ожидание — `ggml_backend_synchronize`, как у Adept: вызовы
  блока идут в очередь, ожидание одно на блок. ggml записывает command buffer заново при каждом вызове, эта
  работа CPU входит в `cpu_ms` и `wall_ms`. Если устройство не поддерживает асинхронный режим, ggml сам ждёт
  конца каждого вызова.
- Устройство: первая дискретная GPU из списка ggml, потом встроенная. Выбрать вручную —
  `GGML_VK_VISIBLE_DEVICES=<номер>`. Список устройств и их возможности (`fp16`, `matrix cores`) ggml
  печатает в stderr при старте.
- На NVIDIA ggml берёт тензорные ядра через cooperative matrix. Без них, для сравнения с `cublas`:
  `GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1`.
- `gpu_ms` нет (`nan`): ggml не отдаёт timestamp queries наружу. Время каждой операции на GPU ggml печатает
  сам при `GGML_VK_PERF_LOGGER=1`.

### cuBLAS

Нужен CUDA Toolkit 11.4 или новее. Если CMake находит его, ядра собираются сами, если нет — проект
собирается без них. Явно: `-DGEMM_WITH_CUBLAS=ON` (без toolkit — ошибка CMake) или `OFF`.
Toolkit не на стандартном месте:

```bash
cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda
```

- Считает первая видимая GPU. Выбрать другую: `CUDA_VISIBLE_DEVICES=<номер>` (номера показывает
  `nvidia-smi -L`). В stderr: `cuda: <имя GPU> (sm_89)`. Нет устройства — ядра пропускаются, причина в stderr.
- Весь пакет — один вызов `cublasGemmStridedBatchedEx`, алгоритм `CUBLAS_GEMM_DEFAULT`. cuBLAS хранит
  матрицы по столбцам, поэтому ядро считает Cᵀ = Bᵀ·Aᵀ: A и B меняются местами в вызове, без копий.
  Общая B — шаг 0 между матрицами пакета.
- Вызов асинхронный, ожидание — `cudaDeviceSynchronize`. `gpu_ms` — CUDA events в начале и в конце блока
  вызовов, как timestamp queries у Adept. Буферы выделяются и заполняются в `prepare()`.

Два ядра фиксируют тип вычислений (compute type), чтобы cuBLAS не выбирал его сам:

| Вход / накопление | `cublas` | `cublas_tc` |
|---|---|---|
| f32 / f32 | `CUBLAS_COMPUTE_32F_PEDANTIC` | нет |
| f16 / f32 | `CUBLAS_COMPUTE_32F_PEDANTIC` | `CUBLAS_COMPUTE_32F` |
| f16 / f16 | `CUBLAS_COMPUTE_16F_PEDANTIC` | `CUBLAS_COMPUTE_16F` |

- `cublas_tc` — режим по умолчанию: cuBLAS использует тензорные ядра, где может.
- `cublas` — pedantic: все шаги строго в типе накопления. Для `16F_PEDANTIC` документация cuBLAS прямо
  говорит, что тензорные ядра выключены, для `32F_PEDANTIC` — нет. Какие CUDA-ядра
  запустились, видно в Nsight Systems: `nsys profile --stats=true ./build/gemm_bench ...`.
- f32 считает только `cublas`: без TF32 и других упрощений точности.
- f16 с накоплением в f16 нужна GPU sm_53+.

Частоты GPU плавают из-за boost и нагрева, поэтому перед замерами на NVIDIA их фиксируют. Это действует
и на Vulkan, в том числе на `op_bench` Adept. Поддерживаемые частоты:

```bash
nvidia-smi --query-supported-clocks=graphics,memory --format=csv
```

Зафиксировать частоту ядра GPU (на Linux заодно включить persistence mode, на Windows — без `sudo`
из консоли администратора):

```bash
sudo nvidia-smi --persistence-mode=1
```

```bash
sudo nvidia-smi --lock-gpu-clocks=<МГц>,<МГц>
```

Частоту брать такую, которую GPU держит под нагрузкой: во время замера `clocks.sm` должна стоять на ней.

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1
```

Частоту памяти фиксирует `--lock-memory-clocks=<МГц>,<МГц>` (поддерживают не все GPU). После замеров:

```bash
sudo nvidia-smi --reset-gpu-clocks
```

## Запуск

Тесты корректности и короткий прогон `gemm_bench`:

```bash
ctest --test-dir build --output-on-failure
```

Все ядра на всех случаях, отчёт в `results/gemm_bench_latest.csv`:

```bash
./build/gemm_bench
```

Вывод — как у `op_bench`: строка `device:` на каждое ядро, затем строка `RESULT` на каждый замер.

```
device: ggml_cpu cpu Apple M1 Pro
device: ggml_vulkan vulkan Apple M1 Pro
RESULT case=square_2048 phase=matmul shape=1x2048x2048x2048 wall_ms=27.5081521 gpu_ms=nan cpu_ms=27.5081521 kernel=ggml_cpu
RESULT case=square_2048_f16_f16 phase=matmul shape=1x2048x2048x2048 wall_ms=6.07085625 gpu_ms=nan cpu_ms=0.00927295 kernel=ggml_vulkan
report: results/gemm_bench_latest.csv
```

Только GPU-ядра на квадратных формах, свой файл отчёта:

```bash
./build/gemm_bench --device gpu --case square --report results/square_gpu.csv
```

| Опция | Что задаёт | По умолчанию |
|---|---|---|
| `--device cpu\|gpu` | только CPU-ядра или только GPU-ядра | все |
| `--kernel a,b` | ядра по имени | все |
| `--iters n` | вызовов в замеряемом блоке | `20` |
| `--warmup n` | холостых вызовов перед замером | `5` |
| `--rounds n` | замеряемых блоков; в отчёт идут медианы | `5` |
| `--case s` | группы случаев, в имени группы есть `s`: `square`, `inference`, `batch`, `matmul` | все |
| `--case-name s` | случаи, в имени которых есть `s` | все |
| `--report path` | файл CSV | `results/gemm_bench_latest.csv` |
| `--list` | ядра и случаи | |

Как у `op_bench`, при любом фильтре (`--device`, `--kernel`, `--case`, `--case-name`) отчёт пишется, только
если задан `--report`: частичный прогон не затирает полный. Код возврата 1 — ядро не прошло проверку точности,
упало или не замерено ни одного случая.

### Сравнение с Adept

Vulkan-ядра Adept меряет `op_bench` из репозитория Adept (`benchmarks/`, собирается с `WITH_BENCHMARKS`).
Чтобы сравнение было честным, обе программы запускают на одной машине с одинаковыми `--warmup`, `--iters`,
`--rounds` и одинаковыми именами и формами случаев. У `op_bench` формы GEMM задаются в
`benchmarks/bench_matmul.cpp`, их имена и формы должны совпадать со списком `bench/cases.cpp`.

```bash
<adept>/build-Release/benchmarks/op_bench --case matmul --report adept.csv
```

```bash
./build/gemm_bench --report gemm_bench.csv
```

График. Нужен `matplotlib`; его можно поставить в `.venv` проекта:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install matplotlib
```

```bash
.venv/bin/python scripts/plot.py gemm_bench.csv adept.csv -o compare.png
```

`plot.py` берёт строки с `phase=matmul`, считает `GOP/s = 2·B·M·K·N / время` и строит график на каждую
группу случаев и пару типов. Линия — ядро; если файлов несколько, к имени ядра добавляется имя файла
(`adept: matmul`). `--time gpu` — GOP/s по `gpu_ms` вместо `wall_ms`, `--log` — логарифмическая шкала.
В stdout печатается GOP/s каждой точки.

## Что замеряется

Замер повторяет `adept::benchmark::measure` (`include/adept/benchmark/measure.hpp` в Adept):

1. `--warmup` холостых вызовов, затем ожидание конца работы на устройстве.
2. `--rounds` раундов. В каждом: ожидание, начало таймера GPU, `--iters` вызовов подряд, конец таймера GPU
   (он сам ждёт устройство) или ожидание.
3. На раунд — время на один вызов: время раунда / `--iters`. В отчёт идёт медиана по раундам.

| Колонка | Что это |
|---|---|
| `case` | имя случая; f16 — с суффиксом `_f16_f32` или `_f16_f16` |
| `phase` | всегда `matmul`, как у GEMM в `op_bench` |
| `shape` | `BxMxKxN`, как у `op_bench`: размер пакета, M, K, N |
| `wall_ms` | полное время на вызов: от первого вызова блока до конца работы на устройстве |
| `gpu_ms` | время на GPU на вызов: от начала до конца блока по таймеру GPU, вместе с простоями между вызовами; `nan`, если ядро его не даёт |
| `cpu_ms` | время, пока CPU отправлял вызовы блока, на вызов |
| `kernel` | ядро |

- У CPU-ядер вызов синхронный: `wall_ms` ≈ `cpu_ms`, `gpu_ms` = `nan`.
- У `ggml_vulkan` и `cublas` вызов только ставит работу в очередь, ожидание одно на блок.
- `gpu_ms` есть только у `cublas` и `cublas_tc` (CUDA events).
- Перед замером каждое ядро один раз считает задачу, результат сверяется с эталоном в double по 1024
  случайным элементам C. Ядро, не прошедшее проверку, не замеряется: в stderr — строка `FAILED`.
- Данные и экземпляр ядра готовятся до замера (`prepare()`): загрузка на устройство, упаковка B не входят во
  время.

## Случаи

| Группа | Случаи | Форма M×K×N, пакет |
|---|---|---|
| `square` | `square_256`, `square_1024`, `square_2048`, `square_4096` | N×N×N |
| `inference` | `inference_m1`, `inference_m8`, `inference_m32`, `inference_m128` | M×4096×4096 |
| `inference` | `inference_k64` | 4096×64×4096 |
| `batch` | `batch_<N>x<B>` для N = 32, 64, 128 и B = 1, 16, 1024 | N×N×N, пакет B |
| `batch` | `batch_64x1024_shared_b` | 64×64×64, пакет 1024, одна B на весь пакет |
| `matmul` | `matmul_attention` | 100×32×100, пакет 8; тот же случай, что в `op_bench` |

Каждая форма есть в трёх парах типов: f32/f32 (без суффикса), f16/f32 (`_f16_f32`), f16/f16 (`_f16_f16`).
Раскладка везде `nn`. Ядро считает только те случаи, которые проходят его `supports()`. Полный список с
формами: `./build/gemm_bench --list`.

## Типы данных

Задача задаётся парой «тип входа / тип накопления», например `f16/f32`. Тип C совпадает с типом
накопления.

| Вход | Накопление и C | Допуск |
|---|---|---|
| f32 | f32 | `(K + 2)·2⁻²⁴` |
| f16 | f32 | `(K + 2)·2⁻²⁴` |
| f16 | f16 | `(K + 2)·2⁻¹¹` |

- f16-входы генерируются сразу представимыми в f16, поэтому эталон в double остаётся точным.
- f16 хранится как `Half` из `half.hpp`; перевод float ↔ half свой, без `_Float16`.
- TF32 не используется. i8 и транспонированных матриц нет: в Adept GEMM для i8 нет, случаи `op_bench` — `nn`.
- `ggml_vulkan` — только f16, `ggml_cpu` и `xnnpack` — только f32, cuBLAS — все пары.

## Профилирование

Под профилировщиком гоняется один случай одного ядра, число вызовов задают `--warmup`, `--iters`, `--rounds`:

```bash
./build/gemm_bench --kernel ggml_vulkan --case-name square_2048_f16_f16 --warmup 5 --iters 20 --rounds 1
```

### NVIDIA

- Nsight Systems — таймлайн: отправки, dispatch-и на GPU, CUDA-ядра cuBLAS. Названия трассировок зависят от
  версии nsys, список — в `nsys profile --help`.

```bash
nsys profile --trace=vulkan,cuda,osrt -o gemm ./build/gemm_bench --kernel cublas --case-name square_2048
```

```bash
nsys stats gemm.nsys-rep
```

- Nsight Graphics — счётчики одного dispatch. GPU Trace с Capture Type «One-shot»: у bench нет кадров и
  present, обычный захват кадра не сработает.
- `cublas`: по именам CUDA-ядер в `nsys stats` видно, взял ли cuBLAS тензорные ядра (`cublas` против `cublas_tc`).
- `ggml_vulkan`: свои метки операций ggml ставит при `GGML_VK_DEBUG_MARKERS=1`, время каждой операции
  печатает при `GGML_VK_PERF_LOGGER=1`.

### M1 Pro (MoltenVK → Metal)

MoltenVK пишет захват Metal в файл, Xcode его открывает. Захват идёт всё время жизни `VkDevice`, поэтому
прогон должен быть коротким. Существующий файл MoltenVK не перезаписывает.

```bash
rm -rf ggml.gputrace
```

```bash
METAL_CAPTURE_ENABLED=1 MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=1 MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=$PWD/ggml.gputrace ./build/gemm_bench --kernel ggml_vulkan --case-name square_1024_f16_f16 --warmup 1 --iters 2 --rounds 1
```

```bash
open ggml.gputrace
```

Открыть захват можно только в Xcode, Command Line Tools не хватит. Счётчиков в файле нет: их снимает Xcode, когда
выполняет записанные команды заново. Для этого на странице Summary в блоке Performance нажать Profile. Где искать
счётчики и стоимость строк шейдера — в [docs/profiling.md](docs/profiling.md).

Счётчики GPU без GUI записывает Instruments из командной строки (`xctrace`, входит в Xcode). `xcode-select`
переключать не нужно: Xcode задаётся через `DEVELOPER_DIR`.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace record --no-prompt --template "Metal System Trace" --instrument "Metal GPU Counters" --output ggml.trace --launch -- $PWD/build/gemm_bench --kernel ggml_vulkan --case-name square_2048_f16_f16 --warmup 100 --iters 20 --rounds 5
```

```bash
python3 scripts/gpu_counters.py --state Maximum ggml.trace
```

`gpu_counters.py` усредняет счётчики по времени работы каждого compute pipeline. Кроме того, он печатает, в каком
состоянии производительности был GPU и какую долю времени ядра GPU выполнял работу других процессов. Pipeline
называются `main0 (<номер>)`. Перед записью нужно усыпить дисплей и дать прогрев не меньше 1 с (`--warmup`
побольше), иначе счётчики искажены. Условия записи, какие счётчики смотреть и как их проверяли — в
[docs/profiling.md](docs/profiling.md).

Пики устройств и замер пропускной способности памяти — в [docs/devices.md](docs/devices.md).

## Как добавить ядро

1. Создать `src/kernels/<name>.cpp`: наследники `GemmKernel` и `GemmInstance`
   и фабрика `std::unique_ptr<GemmKernel> make<Name>Kernel()`.
2. В `device()` вернуть устройство: у CPU-ядер `cpuName()` из `device.hpp`, у GPU-ядер имя GPU.
   В `backend()` — `cpu` у CPU-ядер (по нему работает `--device`). В `supports()` отсечь неподдерживаемые
   типы и формы.
3. В экземпляре: `run()` — один вызов; если он асинхронный, `sync()` ждёт конца всех поставленных вызовов, а
   `readC()` сначала вызывает `sync()`. Если ядро меряет время на GPU: в `beginGpuTiming()` поставить отметку
   начала, в `endGpuTimingMs()` — отметку конца, дождаться её и вернуть время блока в мс. Без них
   `endGpuTimingMs()` просто вызывает `sync()` и возвращает `nan`.
4. Добавить фабрику в `src/kernels/registry.cpp` и файл в `gemm_core` в `CMakeLists.txt`.
5. Ядро подключается опцией CMake `GEMM_WITH_<NAME>`: она добавляет файл ядра, библиотеку и define, а в
   `registry.cpp` фабрика стоит под `#ifdef GEMM_WITH_<NAME>`. Опцию добавить и в проверку `no kernels`.
   Пример — `xnnpack`.
