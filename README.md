# PostgreSQL with JIT compiler for expressions

This is a modified version of [PostgreSQL](https://www.postgresql.org/) with just-in-time compiler for expressions. Current base version is 9.6.1.
Just-in-time compiler (JIT for short) improves database performance significantly on OLAP queries. This version uses LLVM JIT for query compilation and shows up to 20% improvement on TPC-H tests.

## Dependencies

- LLVM version 3.7

## Building

The easiest way is to follow the [process of building vanilla PostgreSQL](https://www.postgresql.org/docs/9.6/static/install-procedure.html). LLVM JIT compiler for expressions is enabled by default.

The most basic procedure is as follows:

```
$ git clone https://github.com/ispras/postgres
$ cd postgres
$ ./configure [configure options]
$ make [make options] -jN
```

### Configure options

option | description
:----: | -----------
`--disable-llvm-jit` | Disables expression compilation with LLVM JIT.
`--with-llvm-config=<llvm_config>` | Specify your own `llvm-config`, version 3.7.0 or 3.7.1 required for build.

### Make options

option | description
:----: | -----------
`LLVM_BACKEND_FILE_LIMIT=<N>` | Parallelize compilation by splitting LLVM backend file, N — number of functions in each file (50 by default).

## PostgreSQL GUC settings

setting | description
:-----: | -----------
`enable_llvm_jit` | Enable LLVM JIT of expressions (enabled by default).
`enable_llvm_dump` | Dump compiled LLVM IR (developer option, disabled by default).
`debug_llvm_jit` | Disable optimization of generated LLVM modules (developer option, disabled by default).

## Internals

Internally, evaluation of each individual expression in PostgreSQL happens by means of calling a function pointer which is stored in the expression object and initialized during expression initialization phase with the address of the corresponding execution function.

Expression JIT presented here works by hooking into expression initialization phase and replacing this pointer with address of a function generated for the expression at run time.

## Advice for developers

- Enabling assertions (configure with `--enable-cassert` option) also turns on LLVM module verification.
- Set `enable_llvm_dump` and `debug_llvm_jit` to on to explore compiled code.

LLVM dumps are stored in the `llvm_dump` subdirectory in the database directory. Each compiled expression is written into three files (substitute `###` with expression number):

- `expr.###` — expression tree, in the “pretty-print” format.
- `dump.###.ll`, `dump.###.opt.ll` — [LLVM assembly](http://llvm.org/docs/LangRef.html) files generated for expression.

## Resources

- LLVM Cauldron 2016: Speeding up query execution in PostgreSQL using LLVM JIT ([slides](http://llvm.org/devmtg/2016-09/slides/Melnik-PostgreSQLLLVM.pdf))
- [Dynamic compilation of expressions in SQL queries for PostgreSQL](http://www.ispras.ru/en/proceedings/isp_28_2016_4/isp_28_2016_4_217/) (paper in Russian, abstract in English)
- PgConf.Russia 2016: Speeding up query execution in PostgreSQL using LLVM JIT compiler ([slides](https://pgconf.ru/en/2016/89652) in Russian)

## License

[PostgreSQL License](https://www.postgresql.org/about/licence/)

