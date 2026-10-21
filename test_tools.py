from tools_builtin import build_registry

reg = build_registry()

print("1. Normal call:")
print("  ", reg.call("calculator", "analysis", expression="(12 + 8) * 3 / 4"))

print("2. Malicious expression:")
print("  ", reg.call("calculator", "analysis", expression="__import__('os').listdir('.')"))

print("3. Wrong specialist (writing may not use calculator):")
print("  ", reg.call("calculator", "writing", expression="1+1"))

print("4. Write then read a file:")
print("  ", reg.call("write_file", "writing", filename="notes.txt", content="Vector DBs store embeddings."))
print("  ", reg.call("read_file", "analysis", filename="notes.txt"))

print("5. Path escape attempt:")
print("  ", reg.call("read_file", "analysis", filename="../secret.txt"))

print("6. Unknown tool:")
print("  ", reg.call("hack_the_planet", "code"))

print("7. Rate limit (calling calculator 35 times):")
failures = 0
for _ in range(35):
    if not reg.call("calculator", "analysis", expression="2+2").success:
        failures += 1
print(f"   {failures} calls were blocked")

print("\nTools per specialist:")
for s in ["research", "analysis", "writing", "code"]:
    print(f"  {s}: {[t.name for t in reg.tools_for(s)]}")

print("\nFirst 6 log entries:")
for entry in reg.log[:6]:
    print(" ", entry)