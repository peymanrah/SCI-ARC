"""
Legacy SCI-ARC/CISL Components

These are the original SCI-ARC (Structural Causal Invariance for ARC) components
that have been superseded by the RLAN architecture.

Structure:
- models/ - Legacy model components (SCIARC, StructuralEncoder, ContentEncoder, etc.)
- training/ - Legacy loss functions (CISL, CICL, SCL)
- scripts/ - Legacy training and evaluation scripts
- tests/ - Legacy test files
- configs/ - Legacy configuration files

To use legacy components:
    import sys
    sys.path.insert(0, 'path/to/others')
    from models.sci_arc import SCIARC, SCIARCConfig
"""
