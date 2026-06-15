# PyEPQ - Java to Python conversion of NIST's DTSA-II EPQ Library
This project encompasses the agent-driven refactoring and validation of the EPQ library from Java to Python.

## Key Components
### Agent Directives
- /docs/PROMPTS.md - Prompt library for generating code specification, code conversion, parity harness, port repair.
- /docs/CONVERSION_GUIDE.md - Supplementary instructions for code specification and code conversion.
- /docs/TESTING_GUIDE.md - Supplementary instructions for parity harness generation.
- /docs/BUG_GUIDE.md - Supplementary instructions for in line bug documentation.

### Project Management
- /`<Class>`/'<Class>'_LEDGER.md - Tracks the progress of code conversion and validation steps across the `<Class>` directory.
- /`<Class>`/BUG_LEDGER.md - Compilation of all intra-file bug ledgers.

### Tools and Scripts
- /tools/check_compliance.py - Unified compliance checking script: Executes 'run_parity.py' (all of the individual parity harnesses), traverses the entire API surface of the Python port while comparing to the Java source API, and updates the `<Class>`_LEDGER.md and BUG_LEDGER.md documents accordingly.
- /tools/dependency_map.py - Produces metrics and a map of dependency relationships.
 

  
*Prepared in partial fulfillment of the requirement of the Department of Energy, Office of Science's Science Undergraduate Laboratory Internship Program under the direction of **Amber Coles** at Savannah River National Laboratory.*
