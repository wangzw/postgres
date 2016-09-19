#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <unordered_set>

namespace
{
    static llvm::cl::opt<bool> Verbose("prune-static-vars-verbose");

    #define LOG if (Verbose) std::clog

    // Remove internal global variables, alongside any global variables and
    // internal functions (transitively) using those. Traverse the call graph
    // from callees to callers and stop at external functions, drop their
    // bodies and continue.
    class PruneStaticVarsPass
    {
      public:
        PruneStaticVarsPass(llvm::Module &module)
            : _module(module)
        {}

        void
        run()
        {
            gatherInternalGlobalVariables();
            gatherUserFunctions();
            pruneModule();
        }

      private:
        // Collect initial set of global variables to remove.
        void
        gatherInternalGlobalVariables()
        {
            for (auto global = _module.global_begin();
                 global != _module.global_end(); ++global)
            {
                // Skip declarations.
                if (global->isDeclaration())
                {
                    continue;
                }

                if (global->hasPrivateLinkage() ||
                    global->hasInternalLinkage())
                {
                    // Skip constants with `unnamed_addr` attribute - they are
                    // safe to duplicate.
                    if (global->isConstant() && global->hasUnnamedAddr())
                    {
                        continue;
                    }

                    // Otherwise, plan to remove the variable.
                    _global_variables.insert(global);
                    LOG << "Marking " << global->getName().str()
                        << " for removal ("
                        << (global->hasInternalLinkage()
                            ? "internal" : "private")
                        << (!global->isConstant() ? ", non-constant" : "")
                        << (!global->hasUnnamedAddr()
                            ? ", no unnamed_addr" : "")
                        << ")\n";
                }
                else
                {
                    // Remove the definition.
                    global->setLinkage(llvm::GlobalValue::ExternalLinkage);
                    global->setInitializer(nullptr);
                }
            }
        }

        // Collect sets of internal and external functions to remove.
        void
        gatherUserFunctions()
        {
            std::function<void(const llvm::Value *, const char *)>
                gather_user_functions;
            std::unordered_set<llvm::GlobalVariable *> extra_global_variables;

            auto add_function = [&](llvm::Function *function,
                                    const char *reason)
            {
                auto log = [function, reason]() {
                    LOG << "Removing " << function->getName().str()
                        << " because of " << reason << '\n';
                };

                switch (function->getLinkage())
                {
                    case llvm::GlobalValue::ExternalLinkage:
                        if (_external_functions.insert(function).second)
                        {
                            log();
                        }
                        break;

                    case llvm::GlobalValue::InternalLinkage:
                        if (_internal_functions.insert(function).second)
                        {
                            log();

                            // Include indirect callers as well.
                            gather_user_functions(function, reason);
                        }
                        break;

                    default:
                        assert(false);
                }
            };

            gather_user_functions = [&](const llvm::Value *value,
                                        const char *reason)
            {
                for (const auto &use: value->uses())
                {
                    auto *user = use.getUser();

                    if (auto *instr = llvm::dyn_cast<llvm::Instruction>(user))
                    {
                        add_function(instr->getParent()->getParent(), reason);
                    }
                    else if (auto *global =
                             llvm::dyn_cast<llvm::GlobalVariable>(user))
                    {
                        if (_global_variables.find(global) ==
                            _global_variables.end())
                        {
                            if (extra_global_variables.insert(global).second)
                            {
                                LOG << "Marking " << global->getName().str()
                                    << " for removal because of " << reason
                                    << '\n';
                            }
                        }
                    }
                    else
                    {
                        assert(!llvm::isa<llvm::GlobalValue>(user));
                        gather_user_functions(user, reason);
                    }
                }
            };

            extra_global_variables.swap(_global_variables);

            // Move variables from `extra_global_variables` to
            // `_global_variables`, one at a time.
            while (!extra_global_variables.empty())
            {
                auto *global = *extra_global_variables.begin();
                extra_global_variables.erase(extra_global_variables.begin());
                _global_variables.insert(global);

                gather_user_functions(global, global->getName().data());
            }
        }

        // Remove global variables and functions collected during the pass.
        void
        pruneModule()
        {
            // First, drop function bodies and initializers, so that there
            // are no uses of these variables and functions left.

            for (auto *function: _external_functions)
            {
                function->deleteBody();
            }

            for (auto *function: _internal_functions)
            {
                function->deleteBody();
            }

            for (auto *global: _global_variables)
            {
                global->setInitializer(nullptr);
            }

            // Then, remove internal functions and globals.
            // External functions are to be bound at run time.

            for (auto *function: _internal_functions)
            {
                function->eraseFromParent();
            }

            for (auto *global: _global_variables)
            {
                global->eraseFromParent();
            }
        }

        llvm::Module &_module;
        std::unordered_set<llvm::GlobalVariable *> _global_variables;
        std::unordered_set<llvm::Function *> _internal_functions;
        std::unordered_set<llvm::Function *> _external_functions;
    };

    class PruneStaticVars : public llvm::ModulePass
    {
      public:
        static char ID;

        PruneStaticVars()
            : llvm::ModulePass(ID)
        {}

        bool
        runOnModule(llvm::Module &module) override
        {
            PruneStaticVarsPass(module).run();
            return true;
        }
    };
}

char PruneStaticVars::ID = 0;
static llvm::RegisterPass<PruneStaticVars> X(
    "prune-static-vars", "Prune Static Variables Pass", false, false);
