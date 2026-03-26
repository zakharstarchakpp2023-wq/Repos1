import random
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from deap import base, creator, tools, algorithms


# Генетичне ядро
def create_toolbox(num_cities, city_coords, tourn_size, gene_mutation_indpb):
    """Створює та налаштовує toolbox для TSP."""
    for attr in ("FitnessMin", "Individual"):
        if hasattr(creator, attr):
            delattr(creator, attr)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def calc_dist(individual):
        distance = 0.0
        for i in range(len(individual)):
            c1 = city_coords[individual[i]]
            c2 = city_coords[individual[(i + 1) % len(individual)]]
            distance += np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return (distance,)

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=gene_mutation_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tourn_size)
    toolbox.register("evaluate", calc_dist)
    return toolbox


# Інтерфейс
class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Genetic Algorithm Visualizer")
        self.root.geometry("1550x920")
        self.root.minsize(1280, 780)

        self._setup_style()

        self.is_running = False
        self.city_coords = {}
        self.history = []
        self.best_individual = None
        self.ga_state = {}
        self.last_city_count = None

        self.generation_var = tk.StringVar(value="0 / 0")
        self.best_var = tk.StringVar(value="—")
        self.stagnation_var = tk.StringVar(value="0")
        self.elapsed_var = tk.StringVar(value="0.0 c")
        self.status_var = tk.StringVar(value="Готовий до запуску")

        self.setup_ui()
        self.generate_cities(initial=True)

    def _setup_style(self):
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")

        self.root.configure(bg="#eef3f8")

        style.configure("App.TFrame", background="#eef3f8")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("Header.TLabel", font=("Segoe UI", 22, "bold"), background="#eef3f8", foreground="#18212b")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11), background="#eef3f8", foreground="#5a6672")
        style.configure("Section.TLabelframe", background="#ffffff")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 12, "bold"), foreground="#213041", background="#ffffff")
        style.configure("FieldLabel.TLabel", font=("Segoe UI", 10), background="#ffffff", foreground="#314154")
        style.configure("Value.TLabel", font=("Segoe UI", 16, "bold"), background="#ffffff", foreground="#18212b")
        style.configure("Caption.TLabel", font=("Segoe UI", 10), background="#ffffff", foreground="#6b7785")
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"), background="#ffffff", foreground="#124d8c")
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 10))
        style.configure("Secondary.TButton", font=("Segoe UI", 11), padding=(14, 10))
        style.configure("StatsCard.TFrame", background="#ffffff")
        style.configure("TEntry", padding=6, font=("Segoe UI", 11))
        style.configure("TNotebook", background="#eef3f8", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 8))

    def setup_ui(self):
        main = ttk.Frame(self.root, style="App.TFrame", padding=16)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(header, text="Оптимізація маршруту TSP", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Оновлений UI: зручні параметри, статистика в реальному часі, прогрес і окремий графік збіжності.",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        sidebar = ttk.Frame(main, style="Panel.TFrame", padding=16)
        sidebar.grid(row=1, column=0, sticky="nsw", padx=(0, 14))
        sidebar.columnconfigure(0, weight=1)

        content = ttk.Frame(main, style="App.TFrame")
        content.grid(row=1, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        self._build_parameters_panel(sidebar)
        self._build_actions_panel(sidebar)
        self._build_status_panel(sidebar)
        self._build_stats_panel(sidebar)

        self._build_content_area(content)

    def _build_parameters_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Параметри алгоритму", style="Section.TLabelframe", padding=12)
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)

        fields = [
            ("К-сть міст", "ent_cities", "25"),
            ("Розмір популяції", "ent_pop", "120"),
            ("Макс. поколінь", "ent_gen", "250"),
            ("Ймовірність кросоверу", "ent_cx", "0.75"),
            ("Ймовірність мутації", "ent_mut", "0.20"),
            ("Інтенсивність мутації", "ent_gene_mut", "0.05"),
            ("Розмір турніру", "ent_tourn", "3"),
            ("Зупинка без змін", "ent_patience", "35"),
            ("Seed (необов'язково)", "ent_seed", ""),
        ]

        self.entries = []
        for row, (label, attr, default) in enumerate(fields):
            ttk.Label(panel, text=label, style="FieldLabel.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=6)
            entry = ttk.Entry(panel, width=16)
            entry.grid(row=row, column=1, sticky="ew", pady=6)
            if default:
                entry.insert(0, default)
            setattr(self, attr, entry)
            self.entries.append(entry)

    def _build_actions_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Керування", style="Section.TLabelframe", padding=12)
        panel.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        panel.columnconfigure((0, 1), weight=1)

        self.btn_run = ttk.Button(panel, text="▶ Запустити", command=self.run_ga, style="Primary.TButton")
        self.btn_run.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.btn_stop = ttk.Button(panel, text="■ Стоп", command=self.stop_ga, style="Secondary.TButton", state=tk.DISABLED)
        self.btn_stop.grid(row=1, column=0, sticky="ew", pady=(10, 0), padx=(0, 6))

        self.btn_regen = ttk.Button(panel, text="⟳ Нові міста", command=self.generate_cities, style="Secondary.TButton")
        self.btn_regen.grid(row=1, column=1, sticky="ew", pady=(10, 0), padx=(6, 0))

    def _build_status_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Стан виконання", style="Section.TLabelframe", padding=12)
        panel.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 8))

        footer = ttk.Frame(panel, style="Panel.TFrame")
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure((0, 1), weight=1)

        ttk.Label(footer, text="Покоління", style="Caption.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.generation_var, style="Value.TLabel").grid(row=1, column=0, sticky="w")

        self.city_count_label = ttk.Label(footer, text="Міста: 0", style="Caption.TLabel")
        self.city_count_label.grid(row=1, column=1, sticky="e")

    def _build_stats_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Поточна статистика", style="Section.TLabelframe", padding=12)
        panel.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        panel.columnconfigure((0, 1), weight=1)

        cards = [
            ("Найкраща дистанція", self.best_var),
            ("Стагнація", self.stagnation_var),
            ("Час", self.elapsed_var),
        ]

        for idx, (title, var) in enumerate(cards):
            card = ttk.Frame(panel, style="StatsCard.TFrame", padding=10)
            card.grid(row=idx, column=0, columnspan=2, sticky="ew", pady=4)
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=title, style="Caption.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=var, style="Value.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

    def _build_content_area(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")

        route_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        chart_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        notebook.add(route_tab, text="Маршрут")
        notebook.add(chart_tab, text="Збіжність")

        route_tab.columnconfigure(0, weight=1)
        route_tab.rowconfigure(0, weight=1)
        chart_tab.columnconfigure(0, weight=1)
        chart_tab.rowconfigure(0, weight=1)

        self.route_fig = Figure(figsize=(9, 6), dpi=100)
        self.route_ax = self.route_fig.add_subplot(111)
        self.route_canvas = FigureCanvasTkAgg(self.route_fig, master=route_tab)
        self.route_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.history_fig = Figure(figsize=(9, 6), dpi=100)
        self.history_ax = self.history_fig.add_subplot(111)
        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=chart_tab)
        self.history_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.draw_route()
        self.draw_history()

    # Допоміжні методи
    def set_controls_running(self, running: bool):
        entry_state = tk.DISABLED if running else tk.NORMAL
        for entry in self.entries:
            entry.configure(state=entry_state)

        self.btn_run.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.btn_regen.configure(state=tk.DISABLED if running else tk.NORMAL)

    def _parse_int(self, entry, name, minimum=1):
        try:
            value = int(entry.get())
        except ValueError as exc:
            raise ValueError(f"Поле '{name}' має бути цілим числом.") from exc
        if value < minimum:
            raise ValueError(f"Поле '{name}' має бути не менше {minimum}.")
        return value

    def _parse_float(self, entry, name, min_value=0.0, max_value=1.0):
        try:
            value = float(entry.get())
        except ValueError as exc:
            raise ValueError(f"Поле '{name}' має бути числом.") from exc
        if not (min_value <= value <= max_value):
            raise ValueError(f"Поле '{name}' має бути в межах [{min_value}; {max_value}].")
        return value

    def read_params(self):
        params = {
            "n_cities": self._parse_int(self.ent_cities, "К-сть міст", 5),
            "pop_size": self._parse_int(self.ent_pop, "Розмір популяції", 4),
            "n_gen": self._parse_int(self.ent_gen, "Макс. поколінь", 1),
            "cxpb": self._parse_float(self.ent_cx, "Ймовірність кросоверу"),
            "mutpb": self._parse_float(self.ent_mut, "Ймовірність мутації"),
            "gene_mut": self._parse_float(self.ent_gene_mut, "Інтенсивність мутації"),
            "tourn": self._parse_int(self.ent_tourn, "Розмір турніру", 2),
            "patience": self._parse_int(self.ent_patience, "Зупинка без змін", 1),
        }

        seed_text = self.ent_seed.get().strip()
        params["seed"] = None if seed_text == "" else int(seed_text)

        if params["tourn"] > params["pop_size"]:
            raise ValueError("Розмір турніру не може перевищувати розмір популяції.")

        return params

    def update_summary(self, generation=0, n_gen=0, best_text="—", stagnation=0, elapsed=0.0):
        self.generation_var.set(f"{generation} / {n_gen}")
        self.best_var.set(best_text)
        self.stagnation_var.set(str(stagnation))
        self.elapsed_var.set(f"{elapsed:.1f} c")

    # Візуалізація
    def draw_route(self, individual=None):
        self.route_ax.clear()
        self.route_ax.set_title("Поточний маршрут", fontsize=14, pad=12)
        self.route_ax.set_xlabel("X")
        self.route_ax.set_ylabel("Y")
        self.route_ax.grid(True, alpha=0.25)

        if not self.city_coords:
            self.route_ax.text(0.5, 0.5, "Немає даних для відображення", ha="center", va="center", transform=self.route_ax.transAxes)
            self.route_canvas.draw_idle()
            return

        xs = [coord[0] for coord in self.city_coords.values()]
        ys = [coord[1] for coord in self.city_coords.values()]
        self.route_ax.scatter(xs, ys, s=38)

        if individual is not None:
            route_x = [self.city_coords[i][0] for i in individual] + [self.city_coords[individual[0]][0]]
            route_y = [self.city_coords[i][1] for i in individual] + [self.city_coords[individual[0]][1]]
            self.route_ax.plot(route_x, route_y, linewidth=1.8)
            self.route_ax.scatter(route_x[0], route_y[0], s=90, marker="s")

            distance = individual.fitness.values[0] if individual.fitness.valid else 0.0
            self.route_ax.set_title(f"Поточний маршрут | Дистанція: {distance:.2f}", fontsize=14, pad=12)

        self.route_canvas.draw_idle()

    def draw_history(self):
        self.history_ax.clear()
        self.history_ax.set_title("Графік збіжності", fontsize=14, pad=12)
        self.history_ax.set_xlabel("Покоління")
        self.history_ax.set_ylabel("Найкраща дистанція")
        self.history_ax.grid(True, alpha=0.25)

        if self.history:
            generations = list(range(1, len(self.history) + 1))
            self.history_ax.plot(generations, self.history, linewidth=2.0)
        else:
            self.history_ax.text(0.5, 0.5, "Після запуску тут з'явиться історія покращення", ha="center", va="center", transform=self.history_ax.transAxes)

        self.history_canvas.draw_idle()

    # Керування станом
    def generate_cities(self, initial=False):
        if self.is_running:
            return

        try:
            n_cities = self._parse_int(self.ent_cities, "К-сть міст", 5)
        except ValueError:
            if initial:
                n_cities = 25
            else:
                messagebox.showerror("Помилка", "Спочатку введіть коректну кількість міст.")
                return

        self.city_coords = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(n_cities)}
        self.best_individual = None
        self.history = []
        self.last_city_count = n_cities

        self.city_count_label.configure(text=f"Міста: {n_cities}")
        self.status_var.set("Згенеровано новий набір міст")
        self.update_summary(generation=0, n_gen=0)
        self.progress["value"] = 0
        self.draw_route()
        self.draw_history()

    def stop_ga(self):
        if self.is_running:
            self.is_running = False
            self.status_var.set("Зупинка після поточного кроку...")

    def run_ga(self):
        if self.is_running:
            return

        try:
            params = self.read_params()
        except Exception as exc:
            messagebox.showerror("Помилка параметрів", str(exc))
            return

        if params["seed"] is not None:
            random.seed(params["seed"])
            np.random.seed(params["seed"])

        if self.last_city_count != params["n_cities"] or not self.city_coords:
            self.city_coords = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(params["n_cities"])}
            self.last_city_count = params["n_cities"]
            self.city_count_label.configure(text=f"Міста: {params['n_cities']}")

        toolbox = create_toolbox(
            params["n_cities"],
            self.city_coords,
            params["tourn"],
            params["gene_mut"],
        )

        population = toolbox.population(n=params["pop_size"])
        hof = tools.HallOfFame(1)

        self.ga_state = {
            "params": params,
            "toolbox": toolbox,
            "population": population,
            "hof": hof,
            "generation": 0,
            "best_dist": float("inf"),
            "stagnation": 0,
            "start_time": time.perf_counter(),
        }

        self.history = []
        self.best_individual = None
        self.is_running = True
        self.set_controls_running(True)
        self.progress.configure(maximum=params["n_gen"], value=0)
        self.status_var.set("Еволюція запущена")
        self.update_summary(generation=0, n_gen=params["n_gen"], best_text="—", stagnation=0, elapsed=0.0)
        self.draw_route()
        self.draw_history()
        self.root.after(1, self.evolve_step)

    def evolve_step(self):
        if not self.is_running:
            self._finish_run("Виконання перервано користувачем")
            return

        params = self.ga_state["params"]
        gen = self.ga_state["generation"]

        if gen >= params["n_gen"]:
            self._finish_run("Завершено повністю")
            return

        toolbox = self.ga_state["toolbox"]
        population = self.ga_state["population"]
        hof = self.ga_state["hof"]

        population = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=params["cxpb"],
            mutpb=params["mutpb"],
            ngen=1,
            halloffame=hof,
            verbose=False,
        )[0]

        self.ga_state["population"] = population
        self.ga_state["generation"] += 1

        current_best = hof[0].fitness.values[0]
        if current_best < self.ga_state["best_dist"]:
            self.ga_state["best_dist"] = current_best
            self.ga_state["stagnation"] = 0
        else:
            self.ga_state["stagnation"] += 1

        self.best_individual = hof[0]
        self.history.append(current_best)

        generation_done = self.ga_state["generation"]
        elapsed = time.perf_counter() - self.ga_state["start_time"]
        stagnation = self.ga_state["stagnation"]

        self.progress["value"] = generation_done
        self.status_var.set(f"Еволюція... покоління {generation_done}")
        self.update_summary(
            generation=generation_done,
            n_gen=params["n_gen"],
            best_text=f"{current_best:.2f}",
            stagnation=stagnation,
            elapsed=elapsed,
        )

        if generation_done == 1 or generation_done % 2 == 0 or generation_done == params["n_gen"]:
            self.draw_route(self.best_individual)
            self.draw_history()

        if stagnation >= params["patience"]:
            self._finish_run("Зупинено: досягнуто межі стагнації")
            return

        self.root.after(1, self.evolve_step)

    def _finish_run(self, status_text):
        elapsed = 0.0
        if self.ga_state:
            elapsed = time.perf_counter() - self.ga_state.get("start_time", time.perf_counter())

        self.is_running = False
        self.set_controls_running(False)
        self.status_var.set(status_text)

        if self.best_individual is not None:
            self.draw_route(self.best_individual)
        else:
            self.draw_route()
        self.draw_history()

        params = self.ga_state.get("params", {})
        self.update_summary(
            generation=self.ga_state.get("generation", 0),
            n_gen=params.get("n_gen", 0),
            best_text=(f"{self.ga_state.get('best_dist', 0):.2f}" if self.best_individual is not None else "—"),
            stagnation=self.ga_state.get("stagnation", 0),
            elapsed=elapsed,
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
