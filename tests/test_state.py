"""Tests for state management: ATP_Store, HistoneStore, Genome, Telomere."""

import pytest
from operon_ai.state.metabolism import ATP_Store, MetabolicState, EnergyType
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength
from operon_ai.state.genome import Genome, Gene, GeneType, ExpressionLevel
from operon_ai.state.telomere import Telomere, LifecyclePhase


class TestATPStore:
    """Tests for the ATP_Store (metabolic budget) class."""

    def test_initial_budget(self):
        """ATP_Store initializes with the specified budget."""
        store = ATP_Store(budget=100, silent=True)
        assert store.atp == 100

    def test_consume_success(self):
        """consume() returns True and deducts when sufficient ATP."""
        store = ATP_Store(budget=100, silent=True)
        result = store.consume(cost=30, operation="test")
        assert result is True
        assert store.atp == 70

    def test_consume_insufficient(self):
        """consume() returns False when insufficient ATP."""
        store = ATP_Store(budget=10, silent=True)
        result = store.consume(cost=20, operation="test")
        assert result is False
        assert store.atp == 10  # Budget unchanged

    def test_consume_exact_budget(self):
        """consume() succeeds when cost equals remaining budget."""
        store = ATP_Store(budget=50, silent=True)
        result = store.consume(cost=50, operation="test")
        assert result is True
        assert store.atp == 0

    def test_multiple_consumptions(self):
        """Multiple consume() calls correctly deplete budget."""
        store = ATP_Store(budget=100, silent=True)
        assert store.consume(cost=25, operation="test1") is True
        assert store.consume(cost=25, operation="test2") is True
        assert store.consume(cost=25, operation="test3") is True
        assert store.atp == 25
        assert store.consume(cost=30, operation="test4") is False  # Would exceed
        assert store.atp == 25

    def test_zero_cost_consumption(self):
        """consume() with zero cost succeeds without change."""
        store = ATP_Store(budget=100, silent=True)
        result = store.consume(cost=0, operation="test")
        assert result is True
        assert store.atp == 100

    def test_multi_currency(self):
        """ATP_Store supports multiple energy currencies."""
        store = ATP_Store(budget=100, gtp_budget=50, nadh_reserve=30, silent=True)
        assert store.atp == 100
        assert store.gtp == 50
        assert store.nadh == 30

    def test_gtp_consumption(self):
        """GTP can be consumed separately from ATP."""
        store = ATP_Store(budget=100, gtp_budget=50, silent=True)
        result = store.consume(cost=20, operation="tool", energy_type=EnergyType.GTP)
        assert result is True
        assert store.gtp == 30
        assert store.atp == 100  # ATP unchanged

    def test_nadh_to_atp_conversion(self):
        """NADH can be converted to ATP."""
        store = ATP_Store(budget=100, nadh_reserve=30, silent=True)
        store.consume(cost=80, operation="drain")  # Drain some ATP
        converted = store.convert_nadh_to_atp(20)
        assert converted == 20
        assert store.nadh == 10
        assert store.atp == 40  # 20 remaining + 20 converted

    def test_metabolic_state_transitions(self):
        """Metabolic state changes based on energy levels."""
        store = ATP_Store(budget=100, silent=True)
        assert store.get_state() == MetabolicState.NORMAL

        # Drain to conserving
        store.consume(cost=75, operation="drain")
        assert store.get_state() == MetabolicState.CONSERVING

        # Drain to starving
        store.consume(cost=20, operation="drain")
        assert store.get_state() == MetabolicState.STARVING

    def test_energy_debt(self):
        """Energy debt system allows borrowing with interest."""
        store = ATP_Store(budget=50, max_debt=30, silent=True)
        # Drain most ATP
        store.consume(cost=40, operation="drain")
        assert store.atp == 10
        # Consume more than available with debt allowed
        # Need 30 but only have 10, so deficit of 20 should go to debt
        result = store.consume(cost=30, operation="borrow", allow_debt=True)
        assert result is True
        assert store.atp == 0  # ATP depleted
        assert store.get_debt() == 20  # 30 - 10 = 20 debt

    def test_regeneration(self):
        """Energy can be regenerated."""
        store = ATP_Store(budget=100, silent=True)
        store.consume(cost=50, operation="drain")
        store.regenerate(20)
        assert store.atp == 70

    def test_energy_transfer(self):
        """Energy can be transferred between stores."""
        donor = ATP_Store(budget=100, silent=True)
        # Recipient needs higher max_atp to receive the transfer
        recipient = ATP_Store(budget=100, silent=True)
        recipient.consume(cost=80, operation="drain")  # Start at 20
        assert recipient.atp == 20

        success = donor.transfer_to(recipient, 30)
        assert success is True
        assert donor.atp == 70
        assert recipient.atp == 50  # 20 + 30 = 50 (under max_atp of 100)


class TestHistoneStore:
    """Tests for the HistoneStore (epigenetic memory) class."""

    def test_initial_state_empty(self):
        """HistoneStore initializes with no markers."""
        store = HistoneStore(silent=True)
        stats = store.get_statistics()
        assert stats['total_markers'] == 0

    def test_add_marker(self):
        """add_marker() adds a new marker."""
        store = HistoneStore(silent=True)
        marker_hash = store.add_marker("Avoid SQL injection patterns")
        assert marker_hash is not None
        stats = store.get_statistics()
        assert stats['total_markers'] == 1

    def test_methylate(self):
        """methylate() adds a permanent marker."""
        store = HistoneStore(silent=True)
        marker_hash = store.methylate("Critical safety rule")
        assert marker_hash is not None
        stats = store.get_statistics()
        assert stats['by_type'].get('methylation', 0) == 1

    def test_acetylate(self):
        """acetylate() adds a temporary marker."""
        store = HistoneStore(silent=True)
        marker_hash = store.acetylate("User preference", decay_hours=24)
        assert marker_hash is not None
        stats = store.get_statistics()
        assert stats['by_type'].get('acetylation', 0) == 1

    def test_add_multiple_markers(self):
        """Multiple markers can be added."""
        store = HistoneStore(silent=True)
        store.methylate("Lesson 1")
        store.methylate("Lesson 2")
        store.acetylate("Lesson 3")
        stats = store.get_statistics()
        assert stats['total_markers'] == 3

    def test_retrieve_context_empty(self):
        """retrieve_context() returns empty result when no markers."""
        store = HistoneStore(silent=True)
        result = store.retrieve_context("any query")
        assert len(result.markers) == 0

    def test_retrieve_context_with_markers(self):
        """retrieve_context() returns relevant markers."""
        store = HistoneStore(silent=True)
        store.methylate("Avoid retrying failed API calls", tags=["api"])
        store.methylate("Check for null values", tags=["validation"])

        result = store.retrieve_context("api")
        assert len(result.markers) >= 1

    def test_retrieve_by_tags(self):
        """Markers can be retrieved by tags."""
        store = HistoneStore(silent=True)
        store.methylate("SQL rule", tags=["sql", "security"])
        store.methylate("API rule", tags=["api", "security"])
        store.methylate("Other rule", tags=["misc"])

        result = store.retrieve_context(tags=["security"])
        assert len(result.markers) == 2

    def test_marker_strength_levels(self):
        """Markers support different strength levels."""
        store = HistoneStore(silent=True)
        store.methylate("Weak rule", strength=MarkerStrength.WEAK)
        store.methylate("Strong rule", strength=MarkerStrength.STRONG)

        # Filter by minimum strength
        result = store.retrieve_context(min_strength=MarkerStrength.STRONG)
        assert len(result.markers) == 1

    def test_inheritance(self):
        """Markers can be inherited to child stores."""
        parent = HistoneStore(silent=True)
        parent.methylate("Core rule", strength=MarkerStrength.STRONG)
        parent.acetylate("Temporary rule", strength=MarkerStrength.WEAK)

        child = HistoneStore(silent=True)
        parent.inherit_to(child, min_strength=MarkerStrength.STRONG)

        child_stats = child.get_statistics()
        assert child_stats['total_markers'] == 1


class TestGenome:
    """Tests for the Genome (immutable configuration) class."""

    def test_initial_genes(self):
        """Genome initializes with provided genes."""
        genome = Genome(
            genes=[Gene(name="model", value="gpt-4")],
            silent=True
        )
        assert genome.get_value("model") == "gpt-4"

    def test_add_gene(self):
        """Genes can be added to genome."""
        genome = Genome(silent=True)
        success = genome.add_gene(Gene(name="temperature", value=0.7))
        assert success is True
        assert genome.get_value("temperature") == 0.7

    def test_immutability(self):
        """Genome rejects duplicate genes when mutations disabled."""
        genome = Genome(allow_mutations=False, silent=True)
        genome.add_gene(Gene(name="model", value="gpt-4"))
        success = genome.add_gene(Gene(name="model", value="gpt-3.5"))
        assert success is False
        assert genome.get_value("model") == "gpt-4"

    def test_express(self):
        """express() returns active configuration."""
        genome = Genome(
            genes=[
                Gene(name="model", value="gpt-4"),
                Gene(name="temp", value=0.7),
            ],
            silent=True
        )
        config = genome.express()
        assert config["model"] == "gpt-4"
        assert config["temp"] == 0.7

    def test_gene_silencing(self):
        """Silenced genes are not expressed."""
        genome = Genome(
            genes=[
                Gene(name="model", value="gpt-4"),
                Gene(name="debug", value=True),
            ],
            silent=True
        )
        genome.silence_gene("debug")
        config = genome.express()
        assert "model" in config
        assert "debug" not in config

    def test_replication(self):
        """Genome can be replicated with mutations."""
        parent = Genome(
            genes=[Gene(name="temp", value=0.7)],
            allow_mutations=True,
            silent=True
        )
        child = parent.replicate(mutations={"temp": 0.9})
        assert parent.get_value("temp") == 0.7
        assert child.get_value("temp") == 0.9
        assert child._generation == 1

    def test_gene_types(self):
        """Different gene types are supported."""
        genome = Genome(
            genes=[
                Gene(name="core", value=1, gene_type=GeneType.STRUCTURAL),
                Gene(name="dormant", value=2, gene_type=GeneType.DORMANT),
            ],
            silent=True
        )
        config = genome.express()
        assert "core" in config
        assert "dormant" not in config  # Dormant genes not expressed


class TestTelomere:
    """Tests for the Telomere (lifecycle management) class."""

    def test_initial_phase(self):
        """Telomere starts in NASCENT phase."""
        telomere = Telomere(max_operations=100, silent=True)
        assert telomere.get_phase() == LifecyclePhase.NASCENT

    def test_start_transitions_to_active(self):
        """start() transitions to ACTIVE phase."""
        telomere = Telomere(max_operations=100, silent=True)
        telomere.start()
        assert telomere.get_phase() == LifecyclePhase.ACTIVE

    def test_tick_shortens_telomere(self):
        """tick() reduces telomere length."""
        telomere = Telomere(max_operations=100, silent=True)
        telomere.start()
        initial = telomere.get_status().telomere_length
        telomere.tick(cost=10)
        assert telomere.get_status().telomere_length == initial - 10

    def test_senescence_on_depletion(self):
        """Telomere enters senescence when depleted."""
        telomere = Telomere(max_operations=20, silent=True)
        telomere.start()
        for _ in range(25):  # Exceed max operations
            telomere.tick(cost=1)
        assert telomere.get_phase() == LifecyclePhase.SENESCENT

    def test_renewal(self):
        """Telomere can be renewed."""
        telomere = Telomere(max_operations=100, allow_renewal=True, silent=True)
        telomere.start()
        for _ in range(80):
            telomere.tick()
        telomere.renew(amount=50)
        status = telomere.get_status()
        assert status.telomere_length > 20

    def test_error_tracking(self):
        """Errors are tracked and can trigger senescence."""
        telomere = Telomere(max_operations=1000, error_threshold=5, silent=True)
        telomere.start()
        for _ in range(6):
            telomere.record_error()
        assert telomere.get_phase() == LifecyclePhase.SENESCENT

    def test_is_operational(self):
        """is_operational() returns correct status."""
        telomere = Telomere(max_operations=100, silent=True)
        assert telomere.is_operational() is True
        telomere.terminate()
        assert telomere.is_operational() is False

    def test_statistics(self):
        """Statistics are tracked correctly."""
        telomere = Telomere(max_operations=100, silent=True)
        telomere.start()
        for _ in range(10):
            telomere.tick()
        stats = telomere.get_statistics()
        assert stats['operations_count'] == 10
        assert stats['phase'] == 'active'
