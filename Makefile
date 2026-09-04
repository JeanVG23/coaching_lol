# Makefile — points d'entrée du projet.
#
# `make demo` est la cible qui rend le dépôt exécutable après un simple clone :
# `data/` est gitignoré, mais `tests/fixtures/demo/` embarque 49 parties réelles
# pseudonymisées. La démo les copie dans un répertoire jetable, puis rejoue les
# scripts de PRODUCTION dessus. Rien n'est simulé sauf l'appel au modèle.

PY := poetry run python3
DEMO_DIR ?= .demo
FIXTURES := tests/fixtures/demo
DEMO_ENV := COACHING_DATA_DIR=$(abspath $(DEMO_DIR))

.PHONY: help demo demo-clean test lint fixtures

help:
	@echo "make demo    — pipeline complet sur les fixtures versionnées (0 réseau, 0 clé)"
	@echo "make test    — pytest + vitest"
	@echo "make lint    — ruff + typecheck TypeScript"
	@echo "make fixtures— régénère tests/fixtures/demo depuis les données locales"

demo: demo-clean
	@echo "\n=== 0/5  Jeu de démo : 49 parties réelles pseudonymisées ==="
	@mkdir -p $(DEMO_DIR)
	@cp -R $(FIXTURES)/. $(DEMO_DIR)/
	@cat $(FIXTURES)/MANIFEST.json
	@echo "\n=== 1/5  Silver : ré-extraction depuis le raw (0 appel API) ==="
	@$(DEMO_ENV) $(PY) src/pipeline_ops/reextract_silver.py
	@echo "\n=== 2/5  Gold : agrégats benchmarkés (facettes win/loss) ==="
	@$(DEMO_ENV) $(PY) src/pipeline_ops/rebuild_gold.py
	@echo "\n=== 3/5  Verdict heuristique : perso vs challenger, à issue égale ==="
	@$(DEMO_ENV) $(PY) src/reporting/compare.py --scope adc --outcome loss
	@echo "\n=== 4/5  Coaching : payload -> schéma -> review (modèle bouchonné) ==="
	@$(DEMO_ENV) $(PY) src/04_coaching/coach.py --scope adc --game --mock-llm
	@echo "\n=== 5/5  Évaluation automatique : ancrage de la review produite ==="
	@$(DEMO_ENV) $(PY) src/04_coaching/grounding.py --player spadzze
	@echo "\nDonnées de la démo dans $(DEMO_DIR)/ (jetable). Vos données réelles"
	@echo "vivent dans data/ et n'ont pas été touchées."

demo-clean:
	@rm -rf $(DEMO_DIR)

# Régénération des fixtures : nécessite le cache raw local, donc réservée au
# poste qui collecte. L'audit de pseudonymisation tourne à la fin.
fixtures:
	@$(PY) src/pipeline_ops/build_demo_fixtures.py

test:
	@$(PY) -m pytest tests/ -q
	@npm test --prefix web/cf

lint:
	@poetry run ruff check .
	@npm run --prefix web/cf typecheck
